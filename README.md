# MiniMax-Remover Nodes for Griptape

This library provides Griptape Nodes for [MiniMax-Remover](https://github.com/zibojia/MiniMax-Remover), a fast and efficient diffusion-based model for removing objects from videos. Provide a source video and a matching mask video, and the model paints out the masked region across every frame, all within your Griptape workflows.

The model runs locally on your own hardware and requires a CUDA-capable GPU.

## Features

- **AI-powered video object removal**: Seamlessly remove objects, people, or artifacts from a video using mask guidance.
- **Diffusion-based inpainting**: Uses the MiniMax-Remover diffusion pipeline for natural-looking fills.
- **Configurable quality/speed tradeoff**: Tune the number of denoising steps to balance quality against processing time.
- **Flexible output resolution**: Control output width and height independently.
- **Local execution**: Model weights are downloaded from Hugging Face and run on your own GPU.

## Requirements

- A CUDA-capable NVIDIA GPU.
- The MiniMax-Remover model weights (`zibojia/minimax-remover`) are downloaded automatically from Hugging Face on first use.

## Installation

1. Clone this repository into your Griptape Nodes workspace directory:

```bash
# Navigate to your workspace directory
# On Mac or Linux you can use the command below to print your workspace directory
cd $(gtn config show | grep workspace_directory | cut -d'"' -f4)
# On Windows, the default workspace directory is a directory named GriptapeNodes in your home directory.
# Usually this is C:\Users\<username>\GriptapeNodes

# Clone the repository (with submodules)
git clone --recurse-submodules https://github.com/griptape-ai/griptape-nodes-minimax-remover-library.git
```

2. Install dependencies:

```bash
cd griptape-nodes-minimax-remover-library
uv sync
```

## Add your library to your installed Engine!

If you haven't already installed your Griptape Nodes engine, follow the installation steps [HERE](https://github.com/griptape-ai/griptape-nodes).
After you've completed those and you have your engine up and running:

1. Copy the path to your `griptape_nodes_library.json` file within the `griptape_nodes_minimax_remover` directory. Right click on the file, and `Copy Path` (Not `Copy Relative Path`).
2. Start up the engine!
3. Navigate to settings.
4. Open your settings and go to the App Events tab. Add an item in **Libraries to Register**.
5. Paste your copied `griptape_nodes_library.json` path from earlier into the new item.
6. Exit out of Settings. It will save automatically!
7. Open up the **Libraries** dropdown on the left sidebar.
8. Your newly registered library should appear! Drag and drop nodes to use them!

## Available Nodes

### MiniMax Video Object Remover

Remove objects from a video using a binary mask, powered by the MiniMax-Remover diffusion model.

- **Input Video**: Source video containing the object to remove.
- **Input Mask**: Binary mask video where white pixels mark the region to remove and black pixels mark the region to keep. Must match the input video frame count.
- **Num Inference Steps**: Number of denoising steps (6-50). More steps yield better quality but slower processing (default: 12).
- **Height**: Output video height in pixels (256-1024, must be a multiple of 8, default: 480).
- **Width**: Output video width in pixels (256-1024, must be a multiple of 8, default: 832).
- **Num Frames**: Number of frames to process (1-81). Must match the input video frame count (default: 81).

Outputs:

- **Output Video**: The processed video with the masked object removed.

## Example Workflow

### Remove an Object from a Video

1. Add a **MiniMax Video Object Remover** node.
2. Connect your source video to the **Input Video** input.
3. Connect a matching binary mask video to the **Input Mask** input (white = remove, black = keep). Make sure the mask frame count matches the source video.
4. Adjust **Num Inference Steps**, **Width**, **Height**, and **Num Frames** as needed.
5. Run the workflow.
6. The result is available on the **Output Video** output.

## Troubleshooting

**"CUDA not available"**
- MiniMax-Remover requires a CUDA-capable NVIDIA GPU. Confirm your drivers and CUDA runtime are installed.

**Frame count mismatch**
- The mask video and source video must have the same number of frames. Set **Num Frames** to match both inputs.

**Slow first run**
- The model weights are downloaded from Hugging Face on first use and cached locally. Subsequent runs are faster.

**Poor removal quality**
- Increase **Num Inference Steps** for higher quality, and ensure your mask cleanly covers the object you want removed.

## License

The Griptape Nodes integration code in this repository is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details. The underlying [MiniMax-Remover](https://github.com/zibojia/MiniMax-Remover) model is subject to its own license terms.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
