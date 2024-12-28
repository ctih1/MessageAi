# MessageAI

## Requirements
Python 3.10
requirements.txt's dependencies

### For CUDA
CUDNN 8.1
CUDA Toolkit 11.2

## Args

`--easy-setup`: Guides the user through training and configuring model. Also checks GPU power, and whether the GPU can be used for accelerated workloads


`--skip-extract`: Used during  `--easy-setup`. Skips extracting messages, which assumes you already have a list of sentences ready


`--ignore-from <array of strings>`: Used during `--easy-setup`. Ignores certain chats while extracting. Uses JSON style array syntax. For example, if we were to ignore people "Mathew" and "James", we'd use `src/message_ai_nevalaonni/__init__.py --easy-setup --ignore-from ["Mathew","James"]`. If you're confused, ask ChatGPT


`--cont-training`: Adds more iterations (epoch) to your model, aka forces it to learn based off of the sentences you've already given it


`--add-training`: Used for adding more data (sentences) to the model


`--evaluate`: Evaluates the model's accuracy


`--local`: Runs the AI without a discord bot
