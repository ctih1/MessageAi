# MessageAI

## Requirements
Python 3.10
requirements.txt's dependencies

### For CUDA
CUDA Toolkit 11.2 (https://developer.nvidia.com/cuda-11.2.0-download-archive)

CUDNN 8.1 (https://developer.nvidia.com/rdp/cudnn-archive, and scroll down until you see "cuDNN v8.1.0 (January 26th, 2021), for CUDA 11.0,11.1 and 11.2")

If you have an **Apple Silicon Mac**, you might have some luck with `tensorflow-metal`, read more below

## Args

`--easy-setup`: Guides the user through training and configuring model. Also checks GPU power, and whether the GPU can be used for accelerated workloads


`--skip-extract`: Used during  `--easy-setup`. Skips extracting messages, which assumes you already have a list of sentences ready


`--ignore-from <array of strings>`: Used during `--easy-setup`. Ignores certain chats while extracting. Uses JSON style array syntax. For example, if we were to ignore people "Mathew" and "James", we'd use `src/message_ai_nevalaonni/__init__.py --easy-setup --ignore-from ["Mathew","James"]`. If you're confused, ask ChatGPT


`--check-gpu`: Checks if GPU is available.


`--cont-training`: Adds more iterations (epoch) to your model, aka forces it to learn based off of the sentences you've already given it


`--add-training`: Used for adding more data (sentences) to the model


`--evaluate`: Evaluates the model's accuracy


`--local`: Runs the AI without a discord bot


## Apple Silicone users
Since I don't own an Apple Silicon device, everything listed below is highly experimental, and not guaranteed to work. If you find issues, make sure to create an issue request! 

1. Clone this repository
2. Make sure you have Python 3.10 installed (During installation, make sure to check the "add to path" box)
3. Go to the folder where you cloned the project
4. Open requirements.txt
5. Change the first line from `tensorflow-gpu==2.10.0` to `tensorflow-metal==0.6.0`
6. Run setup.sh
