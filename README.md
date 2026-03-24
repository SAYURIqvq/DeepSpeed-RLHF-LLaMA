# DeepSpeed-RLHF-LLaMA

基于人类反馈强化学习（RLHF）对 LLaMA 进行微调。

## 基于 DeepSpeed Chat 的改动

### 第一步（Step 1）

- alpaca_rlhf/deepspeed_chat/training/step1_supervised_finetuning/main.py#main()

- 设置特殊标记（special tokens）

- ![](./figures/modifications/step1/special_tokens.png)

- alpaca_rlhf/deepspeed_chat/training/utils/data/data_utils.py#create_dataset_split()

- 仅在回复部分进行训练，并添加 EOS（结束标记）

- ![](./figures/modifications/step1/train_only_on_responses.png)

- 移除对话结束标记（end_of_conversation_token）

- ![](./figures/modifications/step1/remove_eoc.png)

- alpaca_rlhf/deepspeed_chat/training/utils/data/data_utils.py#PromptDataset#__getitem__

- 标签（labels）与输入（input）不同

- ![](./figures/modifications/step1/lables_differ_from_input.png)

- alpaca_rlhf/deepspeed_chat/training/utils/data/raw_datasets.py#MultiTurnAlpacaDataset

- 新增 MultiTurnAlpacaDataset（多轮对话数据集）

- ![](./figures/modifications/step1/multi_turn_alpaca_dataset.png)

- alpaca_rlhf/deepspeed_chat/training/utils/module/lora.py#convert_linear_layer_to_lora

- 支持 LoRA 多模块名称

- ![](./figures/modifications/step1/multi_lora_part_names.png)

### 第 2 步

- ![](figures/step2/train_reward.png)

- ![](figures/step2/train_reward_diff.png)

- 统计所选回复的奖励值的均值和标准差，并在第 3 步中用于对奖励进行归一化。在一次实验中，这两个值分别为 -0.8677118420600891 和 0.2210693359375，并在 alpaca_rlhf/deepspeed_chat/training/step3_rlhf_finetuning/ppo_trainer.py#DeepSpeedPPOTrainer#generate_experience 方法中使用：

- ![](./figures/modifications/step3/normalize_reward.png)

- step3: sh run.sh --num_gpus 2 /tmp/pycharm_project_227/alpaca_rlhf/deepspeed_chat/training/step3_rlhf_finetuning/main.py --data_output_path /root/autodl-tmp/rlhf/tmp/ --actor_model_name_or_path /root/autodl-tmp/rlhf/actor/ --tokenizer_name_or_path decapoda-research/llama-7b-hf --critic_model_name_or_path /root/autodl-tmp/rlhf/critic --actor_zero_stage 2 --critic_zero_stage 2 --num_padding_at_beginning 0 --per_device_train_batch_size 4 --per_device_mini_train_batch_size 4 --ppo_epochs 2 --actor_learning_rate 9.65e-6 --critic_learning_rate 5e-6 --gradient_accumulation_steps 1 --deepspeed --actor_lora_dim 8 --actor_lora_module_name q_proj --critic_lora_dim 8 --critic_lora_module_name q_proj,k_proj --only_optimize_lora --output_dir /root/autodl-tmp/rlhf/final

- 第 3 步的训练过程

- ![](./figures/step3/train_actor_loss.png)

- ![](./figures/step3/train_cri_loss.png)

- ![](./figures/step3/train_average_reward.png)

- 推理

- nohup sh run_inference.sh 0 alpaca_rlhf/inference/llama_chatbot_gradio.py --path /root/autodl-tmp/rlhf/final/actor > rlhf_inference.log 2>&1 &

- nohup sh run_inference.sh 0 alpaca_rlhf/inference/llama_chatbot_gradio.py --path /root/autodl-tmp/rlhf/actor > sft_inference.log 2>&1 &

## SFT 与 RLHF 的对比

- 对话

- SFT

- ![](figures/step3/chat_sft.png)

- RLHF

- ![](figures/step3/chat_rlhf.png)

- 故事生成

- SFT

- ![](figures/step3/story_sft.png)

- RLHF

- ![](figures/step3/story_rlhf.png)

## 参考

### 文章

- [如何正确复现 Instruct GPT / RLHF?](https://zhuanlan.zhihu.com/p/622134699)

- [影响 PPO 算法性能的 10 个关键技巧（附 PPO 算法简洁 PyTorch 实现）](https://zhuanlan.zhihu.com/p/512327050)

### 来源

- [Awesome RLHF](https://github.com/opendilab/awesome-RLHF)

### 工具

- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)

### 数据集

- [Stanford Human Preferences Dataset (SHP)](https://huggingface.co/datasets/stanfordnlp/SHP)

- [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)

- [hh-rlhf](https://github.com/anthropics/hh-rlhf)

- 基于人类反馈强化学习训练有帮助且无害的助手 [[论文](https://arxiv.org/abs/2204.05862)]

- [Dahoas/static-hh](https://huggingface.co/datasets/Dahoas/static-hh)

- [Dahoas/rm-static](https://huggingface.co/datasets/Dahoas/rm-static)

- GPT-4-LLM

- [GitHub](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)

- [论文](https://arxiv.org/pdf/2304.03277.pdf)

- [网站](https://instruction-tuning-with-gpt-4.github.io/)

- Open-Assistant

- [网站](https://open-assistant.io/zh)

- [GitHub](https://github.com/LAION-AI/Open-Assistant)

- [论文](./papers/2023-OpenAssistant%20Conversations%20-%20Democratizing%20Large%20Language%20Model%20Alignment.pdf)

