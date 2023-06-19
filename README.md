# RM-PRT: Realistic Robotic Manipulation Simulator and Benchmark with Progressive Reasoning Tasks

**RM-PRT** is a Robotic Manipulation with Progressive Reasoning Tasks benchmark based on a realistic robotic manipulation simulator. Specifically, the RM-PRT benchmark builds a new high-fidelity digital twin scene based on Unreal Engine 5, which includes 782 categories, 2023 objects, and 15K natural language instructions generated by ChatGPT for a detailed evaluation of robot manipulation.

![Simulator](./imgs/Simulator.jpg)

## Getting Started
First download a [model checkpoint](https://drive.google.com/file/d/1shH1DV6_rrq7hS6Zn0LrfT7LXbDQt3Us/view?usp=drive_link). <br>
Run the inference code.
```
python inference.py --host=localhost:30001 --action_nums=8
```
