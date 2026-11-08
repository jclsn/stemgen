use std::{
    collections::{BTreeMap, VecDeque},
    ffi::OsStr,
    path::PathBuf,
    sync::{mpsc, Arc, Mutex},
};

use ffmpeg_next::Packet;
use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use stemgen::{
    demucs::{Demucs, DemusOpts, Device},
    nistem::{self, NIStem},
    track::Track,
};

use crate::cli::{Cli, GenerateArgs};

fn split_file_at_dot(file: &OsStr) -> (&OsStr, Option<&OsStr>) {
    let slice = file.as_encoded_bytes();
    if slice == b".." {
        return (file, None);
    }

    let mut current_idx = slice.len();

    while current_idx > 0 {
        match slice[1..current_idx].iter().rposition(|b| *b == b'.') {
            Some(i) =>  {
                let ext = &slice[i+1..].to_ascii_lowercase();
                if !ext.iter().all(|c|(*c >= b'0' && *c <= b'9') || (*c >= b'a' && *c <= b'z') || *c == b'.') {
                    break;
                }
                current_idx = i + 1;
            },
            None => break,
        };
    }

    if current_idx == slice.len() {
        return (file, None)
    }

    let before = &slice[..current_idx];
    let after = &slice[current_idx..];
    unsafe {
        (
            OsStr::from_encoded_bytes_unchecked(before),
            Some(OsStr::from_encoded_bytes_unchecked(after)),
        )
    }
}

pub fn generate(ctx: &Cli, command: &GenerateArgs) -> Result<bool, Box<dyn std::error::Error>> {
    // One model instance per device; they separate different chunks in parallel.
    let devices = if command.device.is_empty() { vec![Device::default()] } else { command.device.clone() };
    let mut demucs = devices
        .iter()
        .map(|device| Demucs::new_from_file(&command.model, DemusOpts { threads: command.thread, device: *device }))
        .collect::<Result<Vec<_>, _>>()?;
    let mut has_failure = false;
    let sample_rate: u64 = ctx.sample_rate.into();

    let mut files: Vec<Result<glob::Paths, glob::PatternError>> = command.files.iter().map(|raw|glob(&raw)).collect();

    if let Some(err) = files.iter().find_map(|r|r.as_ref().err()) {
        return Err(format!("unable to render the glob: {}", err).into())
    }

    let files: Vec<PathBuf> = files.iter_mut().filter_map(|r|r.as_mut().ok()).flatten().filter_map(|r|r.ok()).collect();

    for file in &files {
        let filename = file.file_name().map(split_file_at_dot).and_then(|(before, _after)| Some(before));
        if filename.is_none() {
            eprintln!(
                "Unable to detect filename from {}",
                file.display()
            );
            has_failure |= true;
            continue;
        }
        let filename = filename.unwrap();
        let output_filename = format!(
            "{}.{}",
            filename.display(),
            ctx.ext
        );
        let output_file = command.output.join(output_filename);
        if output_file.exists() {
            if !ctx.force {
                eprintln!(
                    "Cannot proceed with {}: stem file already exist in output directory!",
                    output_file.display()
                );
                has_failure |= true;
                continue;
            }
            std::fs::remove_file(&output_file)?;
        }
        let mut input = Track::new(file)?;
        let mut nistem = if command.preserved_original_as_master {
            NIStem::new_with_preserved_original(&output_file, input.args(), ctx)?
        } else {
            NIStem::new_with_consistent_streams(&output_file, ctx)?
        };
        nistem.clone(file)?;
        let pb = ProgressBar::new(2 * input.total() as u64);
        pb.set_style(
            ProgressStyle::with_template(
                &format!("{{spinner:.green}} {} [{{wide_bar:.cyan/blue}}] [{{elapsed_precise}}] {{percent}}% ({{eta}})", filename.display()),
            )
            .unwrap()
            .progress_chars("#>-"),
        );

        let preserved = matches!(nistem, NIStem::PreservedMaster(..));
        // Each device runs inference on its own worker thread and takes the next chunk as soon as
        // it is free, so chunks are separated in parallel (and while earlier ones are encoded).
        // Results are encoded in chunk order.
        let devices = demucs.len();
        std::thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
            let (job_tx, job_rx) = mpsc::sync_channel::<(usize, Vec<f32>)>(devices);
            let job_rx = Arc::new(Mutex::new(job_rx));
            let (result_tx, result_rx) = mpsc::channel::<(usize, Result<Vec<Vec<f32>>, String>)>();
            // Average seconds per chunk of each device, once it has separated one.
            let speeds: Arc<Mutex<Vec<Option<f64>>>> = Arc::new(Mutex::new(vec![None; devices]));
            for (device, worker) in demucs.iter_mut().enumerate() {
                let (job_rx, result_tx, speeds) = (Arc::clone(&job_rx), result_tx.clone(), Arc::clone(&speeds));
                scope.spawn(move || {
                    let (mut busy, mut chunks) = (0f64, 0u32);
                    loop {
                        let job = job_rx.lock().unwrap().recv();
                        let Ok((index, samples)) = job else { break };
                        let start = std::time::Instant::now();
                        let result = worker.separate(&samples).map_err(|e| e.to_string());
                        (busy, chunks) = (busy + start.elapsed().as_secs_f64(), chunks + 1);
                        if result_tx.send((index, result)).is_err() {
                            break;
                        }
                        // A much slower device mostly delays the in-order encoding (whoever gets the
                        // last chunk determines when the file is done), so it stops taking chunks.
                        let mut speeds = speeds.lock().unwrap();
                        speeds[device] = Some(busy / f64::from(chunks));
                        let fastest = speeds.iter().flatten().copied().fold(f64::INFINITY, f64::min);
                        if busy / f64::from(chunks) > 3.0 * fastest {
                            eprintln!("Device {} is more than 3x slower than the fastest one, no longer using it for this file", device + 1);
                            break;
                        }
                    }
                });
            }
            drop(result_tx);

            let mut buf: Vec<f32> = vec![0f32; Demucs::CHUNK_SAMPLES];
            // Original packets/samples of the chunks not encoded yet, in chunk order.
            let mut originals: VecDeque<(Vec<Packet>, Vec<f32>, usize, bool)> = VecDeque::new();
            let mut separated: BTreeMap<usize, Vec<Vec<f32>>> = BTreeMap::new();
            let (mut submitted, mut next, mut encoded) = (0usize, 0usize, 0usize);
            let mut input_done = false;
            loop {
                // Chunks are encoded in order, so a slower device holding an early chunk must not
                // starve the others: let up to 4 chunks per device be in flight.
                while !input_done && submitted - next < 4 * devices {
                    let mut original_packets = Vec::with_capacity(512);
                    let size = input.read(preserved.then_some(&mut original_packets), &mut buf)?;
                    let last = size != buf.len();
                    let original_buffer = if preserved { Vec::new() } else { buf[..size].to_vec() };
                    job_tx.send((submitted, buf[..size].to_vec()))?;
                    originals.push_back((original_packets, original_buffer, size, last));
                    submitted += 1;
                    input_done = last;
                }
                let Some((original_packets, original_buffer, size, last)) = originals.pop_front() else {
                    break;
                };
                while !separated.contains_key(&next) {
                    let (index, result) = result_rx.recv()?;
                    separated.insert(index, result?);
                }
                let mut data = separated.remove(&next).unwrap();
                next += 1;
                // Progress counts encoded samples: reading runs ahead of separation.
                encoded += size;
                pb.set_position(encoded as u64 / sample_rate);
                if preserved {
                    nistem.write_preserved(original_packets, data)?;
                } else {
                    data.insert(0, original_buffer);
                    nistem.write_consistent(data)?;
                }
                if last {
                    break;
                }
            }
            Ok(())
        })?;

        pb.finish_with_message(format!("downloaded {}", filename.display()));
        nistem.flush(nistem::Atom {
            stems: [
                nistem::AtomStem {
                    color: ctx.drum_stem_color.to_owned(),
                    name: ctx.drum_stem_label.to_owned(),
                },
                nistem::AtomStem {
                    color: ctx.bass_stem_color.to_owned(),
                    name: ctx.bass_stem_label.to_owned(),
                },
                nistem::AtomStem {
                    color: ctx.other_stem_color.to_owned(),
                    name: ctx.other_stem_label.to_owned(),
                },
                nistem::AtomStem {
                    color: ctx.vocal_stem_color.to_owned(),
                    name: ctx.vocal_stem_label.to_owned(),
                },
            ],
            version: 1,
            ..Default::default()
        })?;
    }
    Ok(has_failure)
}

#[cfg(test)]
mod tests {

    use std::path::Path;

    use stemgen::nistem::{Codec, SampleRate};

    use crate::{cli::GenerateArgs, constants::DEFAULT_EXT, generate::{generate, split_file_at_dot}, Cli, Commands};

    #[test]
    fn test_generate_command() {
        let ctx = Cli {
            force: true,
            verbose: false,
            codec: Codec::OPUS,
            sample_rate: SampleRate::Hz48000,
            ext: DEFAULT_EXT.to_owned(),
            command: Commands::Generate(GenerateArgs {
                files: vec!["../testdata/Oddchap - Sound 104.mp3".into()],
                output: "..".into(),
                preserved_original_as_master: false,
                ..Default::default()
            }),
            ..Default::default()
        };
        if let Commands::Generate(command) = &ctx.command {
            let result = generate(&ctx, command);
            assert!(
                matches!(result, Ok(false)),
                "Expected value to match pattern, but got: {result:?}"
            );
        } else {
            unreachable!("unexpected command value")
        }

        let ctx = Cli {
            force: true,
            verbose: false,
            codec: Codec::OPUS,
            sample_rate: SampleRate::Hz48000,
            ext: DEFAULT_EXT.to_owned(),
            command: Commands::Generate(GenerateArgs {
                files: vec!["../**/*.mp3".into()],
                output: "..".into(),
                preserved_original_as_master: false,
                ..Default::default()
            }),
            ..Default::default()
        };
        if let Commands::Generate(command) = &ctx.command {
            let result = generate(&ctx, command);
            assert!(
                matches!(result, Ok(false)),
                "Expected value to match pattern, but got: {result:?}"
            );
        } else {
            unreachable!("unexpected command value")
        }
    }

    #[test]
    fn test_can_get_file_name(){
        let file_name = Path::new("Flo Rida - Low (feat. T-Pain).ogg").file_name().map(split_file_at_dot).and_then(|(before, _after)| Some(before)).unwrap().to_str().unwrap().to_owned();
        assert_eq!(&file_name, "Flo Rida - Low (feat. T-Pain)");

        let file_name = Path::new("Flo Rida - Low (feat. T-Pain).stem.mp4").file_name().map(split_file_at_dot).and_then(|(before, _after)| Some(before)).unwrap().to_str().unwrap().to_owned();
        assert_eq!(&file_name, "Flo Rida - Low (feat. T-Pain)");
    }
}
