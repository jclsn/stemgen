use std::io::Write;
use std::process::exit;

use clap::Parser;

use crate::cli::{Cli, Commands, prepare_ffmpeg};

mod cli;
pub mod constants;
mod create;
mod generate;

fn run(args: &Cli) -> Result<bool, Box<dyn std::error::Error>> {
    match &args.command {
        Commands::Generate(command) => {
            prepare_ffmpeg(args)?;
            generate::generate(args, command)
        }
        Commands::Create(command) => {
            prepare_ffmpeg(args)?;
            create::create(args, command)
        }
    }
}

/// Whether process teardown must be skipped on exit.
///
/// `ort` releases its global ONNX Runtime environment from a `.fini_array` entry in this
/// executable, which the dynamic loader runs *after* the Vulkan loader and driver have
/// already been finalized. With the WebGPU execution provider, releasing the environment
/// destroys the Dawn device, which then calls into the torn-down Vulkan driver and segfaults.
fn skip_teardown(args: &Cli) -> bool {
    match &args.command {
        #[cfg(feature = "webgpu")]
        Commands::Generate(command) => matches!(command.device, stemgen::demucs::Device::WebGPU),
        _ => false,
    }
}

fn main() {
    let args = Cli::parse();

    let code = match run(&args) {
        Ok(false) => 0,
        Ok(true) => 1,
        Err(err) => {
            eprintln!("Error: {err:?}");
            1
        }
    };

    if skip_teardown(&args) {
        // All output files are closed by now; only library teardown is skipped.
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        unsafe { libc::_exit(code) };
    }
    exit(code);
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    use stemgen::{
        demucs::{Device, Model},
        nistem::{Codec, Color, SampleRate},
    };

    use crate::{
        cli::{Commands, CreateArgs, GenerateArgs}, Cli
    };

    #[test]
    fn test_generate_command() {
        let arg_vec = vec![
            "stemgen",
            "--codec",
            "flac",
            "generate",
            "./my_file.mp3",
            "~/MyMusic",
            "--model",
            "http://example.com/htdemucs.onnx",
        ];
        let ctx = Cli::try_parse_from(arg_vec);
        assert!(
            matches!(
                &ctx,
                Ok(Cli {
                    force: false,
                    verbose: false,
                    codec: Codec::FLAC,
                    sample_rate: SampleRate::Hz44100,
                    command: Commands::Generate (GenerateArgs {
                        files,
                        output,
                        device: Device::CPU,
                        model: Model::Url(model_url),
                        thread: 4,
                        preserved_original_as_master: false
                    }),
                    drum_stem_label,
                    bass_stem_label,
                    other_stem_label,
                    vocal_stem_label,
                    drum_stem_color: Color(0x009E73),
                    bass_stem_color: Color(0xD55E00),
                    other_stem_color: Color(0xCC79A7),
                    vocal_stem_color: Color(0x56B4E9),
                    ext
                }) if (
                    drum_stem_label == "Drums" &&
                    bass_stem_label == "Bass" &&
                    other_stem_label == "Other" &&
                    vocal_stem_label == "Vocals" &&
                    ext == "stem.mp4" &&
                    model_url == "http://example.com/htdemucs.onnx" &&
                    output.display().to_string() == "~/MyMusic" &&
                    *files == [<&str as Into<String>>::into("./my_file.mp3")]
                )
            ),
            "Expected value to match pattern, but got: {ctx:?}"
        );
    }

    #[test]
    fn test_create_command() {
        let arg_vec = vec![
            "stemgen", "create",
            "--mastered", "Pre-mastered mix.mp3",
            "--drum", "drum part.mp3",
            "--bass", "bass part.mp3",
            "--other", "other part.mp3",
            "--vocal", "vocal part.mp3",
            "Artist - Title.stem.mp4"
        ];
        let ctx = Cli::try_parse_from(arg_vec);
        assert!(
            matches!(
                &ctx,
                Ok(Cli {
                    force: false,
                    verbose: false,
                    codec: Codec::AAC,
                    sample_rate: SampleRate::Hz44100,
                    command: Commands::Create (CreateArgs {
                        output,
                        mastered,
                        drum,
                        bass,
                        other,
                        vocal,
                        copy_id3tags_from_mastered: true,
                        ..
                    }),
                    drum_stem_label,
                    bass_stem_label,
                    other_stem_label,
                    vocal_stem_label,
                    drum_stem_color: Color(0x009E73),
                    bass_stem_color: Color(0xD55E00),
                    other_stem_color: Color(0xCC79A7),
                    vocal_stem_color: Color(0x56B4E9),
                    ext
                }) if (
                    drum_stem_label == "Drums" &&
                    bass_stem_label == "Bass" &&
                    other_stem_label == "Other" &&
                    vocal_stem_label == "Vocals" &&
                    ext == "stem.mp4" &&
                    output.display().to_string() == "Artist - Title.stem.mp4" &&
                    mastered.display().to_string() == "Pre-mastered mix.mp3" &&
                    drum.display().to_string() == "drum part.mp3" &&
                    bass.display().to_string() == "bass part.mp3" &&
                    other.display().to_string() == "other part.mp3" &&
                    vocal.display().to_string() == "vocal part.mp3"
                )
            ),
            "Expected value to match pattern, but got: {ctx:?}"
        );
    }

    #[test]
    fn test_create_command_with_customized() {
        let arg_vec = vec![

            "stemgen", "create",
                "--mastered", "Pre-mastered mix.mp3",
                "--drum", "Kick part.mp3",
                "--bass", "SubBass part.mp3",
                "--other", "synth part.mp3",
                "--vocal", "Voices part.mp3",
                "--drum-stem-label", "Kick",
                "--drum-stem-color", "#37e4d0",
                "--bass-stem-label", "SubBass",
                "--bass-stem-color", "#656bba",
                "--other-stem-label", "Synths",
                "--other-stem-color", "#52d034",
                "--vocal-stem-label", "Voices",
                "--vocal-stem-color", "#daae2a",
                "Artist - Title.stem.mp4"
        ];
        let ctx = Cli::try_parse_from(arg_vec);
        assert!(
            matches!(
                &ctx,
                Ok(Cli {
                    force: false,
                    verbose: false,
                    codec: Codec::AAC,
                    sample_rate: SampleRate::Hz44100,
                    command: Commands::Create (CreateArgs {
                        output,
                        mastered,
                        drum,
                        bass,
                        other,
                        vocal,
                        copy_id3tags_from_mastered: true,
                        ..
                    }),
                    drum_stem_label,
                    bass_stem_label,
                    other_stem_label,
                    vocal_stem_label,
                    drum_stem_color: Color(0x37e4d0),
                    bass_stem_color: Color(0x656bba),
                    other_stem_color: Color(0x52d034),
                    vocal_stem_color: Color(0xdaae2a),
                    ext
                }) if (
                    drum_stem_label == "Kick" &&
                    bass_stem_label == "SubBass" &&
                    other_stem_label == "Synths" &&
                    vocal_stem_label == "Voices" &&
                    ext == "stem.mp4" &&
                    output.display().to_string() == "Artist - Title.stem.mp4" &&
                    mastered.display().to_string() == "Pre-mastered mix.mp3" &&
                    drum.display().to_string() == "Kick part.mp3" &&
                    bass.display().to_string() == "SubBass part.mp3" &&
                    other.display().to_string() == "synth part.mp3" &&
                    vocal.display().to_string() == "Voices part.mp3"
                )
            ),
            "Expected value to match pattern, but got: {ctx:?}"
        );
    }
}
