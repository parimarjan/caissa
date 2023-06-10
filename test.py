import os
import multiprocess.context as ctx
import pdb

# /home/gridsan/pnegi
# os.environ["HF_DATASETS_CACHE"] = "/home/gridsan/pnegi/hf/"
# os.environ["HF_CACHE"] = "/home/gridsan/pnegi/hf/"
# os.environ["TRANSFORMERS_CACHE"] = "/home/gridsan/pnegi/hf/"

# os.environ["HF_DATASETS_CACHE"] = "/home/gridsan/pnegi/hf/"
# os.environ["TRANSFORMERS_CACHE"] = "/home/gridsan/pnegi/hf/"

if __name__ == "__main__":
    # ctx._force_start_method('fork')
    # ctx._force_start_method('spawn')
    # os.environ["HF_HOME"] = "/state/partition1/user/pnegi/huggingface"
    # os.environ["HF_HOME"] = "/home/gridsan/pnegi/hf2"

    from datasets import load_dataset

    # Load dataset from the hub
    # dataset = load_dataset("samsum", download_mode="reuse_cache_if_exists")
    # dataset = load_dataset("samsum")
    # dataset = load_dataset("cb.csv")
    dataset = load_dataset('csv',
            data_files={'train': "cb.csv",'test': "cb.csv"})

    # pdb.set_trace()
    # DS="/home/gridsan/pnegi/hf/samsum/samsum/0.0.0/f1d7c6b7353e6de335d444e424dc002ef70d1277109031327bc9cc6af5d3d46e"
    # dataset = load_dataset(DS)
    print("loading done!")

    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['test'])}")

    # pdb.set_trace()

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from transformers import AutoModelForCausalLM, AutoModel

    # model_id2="google/flan-t5-xxl"
    # model_id = "huggyllama/llama-7b"
    # MPATH="/state/partition1/user/pnegi/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16"

    model_id = "philschmid/flan-t5-xxl-sharded-fp16"
    # model = AutoModel.from_pretrained(model_id, load_in_8bit=True,
            # device_map="auto")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # model = AutoModel.from_pretrained(MPATH, load_in_8bit=True,
            # device_map="auto")
    # model = AutoModelForCausalLM.from_pretrained(MPATH, load_in_8bit=True,
            # device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained(MPATH)

    # if tokenizer.pad_token is None:
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    ## crashes
    # from transformers import AutoModel
    # model_id = "tiiuae/falcon-7b"
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModel.from_pretrained(model_id,
            # trust_remote_code=True)

    from datasets import concatenate_datasets
    import numpy as np
    # The maximum total input sequence length after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = concatenate_datasets([dataset["train"],
        dataset["test"]]).map(lambda x: tokenizer(x["fens"], truncation=False), batched=True, remove_columns=[])
    input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
    # take 85 percentile of max length for better utilization
    max_source_length = int(np.percentile(input_lenghts, 100))
    print(f"Max source length: {max_source_length}")

    # The maximum total sequence length for target text after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be
    # padded."
    tokenized_targets = concatenate_datasets([dataset["train"],
        dataset["test"]]).map(lambda x: tokenizer(x["comments"],
            truncation=False), batched=True, remove_columns=["fens"])

    target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
    # take 90 percentile of max length for better utilization
    max_target_length = int(np.percentile(target_lenghts, 90))
    print(f"Max target length: {max_target_length}")

    def preprocess_function(sample,padding="max_length"):
        # add prefix to the input for t5
        inputs = ["Describe the chess position: " + item for item in sample["fens"]]

        # tokenize inputs
        model_inputs = tokenizer(inputs, max_length=max_source_length,
                padding=padding, truncation=False)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=sample["comments"],
                max_length=max_target_length, padding=padding, truncation=False)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [ [(l if l != tokenizer.pad_token_id else
                -100) for l in label] for label in labels["input_ids"] ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True,
            remove_columns=[])

    # tokenized_dataset = dataset
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

    # save datasets to disk for later easy loading
    tokenized_dataset["train"].save_to_disk("data/train")
    tokenized_dataset["test"].save_to_disk("data/eval")

    from transformers import AutoModelForSeq2SeqLM

    # huggingface hub model id
    # model_id = "philschmid/flan-t5-xxl-sharded-fp16"

    # load model from the hub
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")

    from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

    # Define LoRA Config
    lora_config = LoraConfig(
     r=16,
     lora_alpha=32,
     target_modules=["q", "v"],
     # target_modules=["q_proj","k_proj", "v_proj", "o_proj"],
     lora_dropout=0.05,
     bias="none",
     task_type=TaskType.SEQ_2_SEQ_LM
     # task_type="CAUSAL_LM",
    )

    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # trainable params: 18874368 || all params: 11154206720 || trainable%: 0.16921300163961817

    from transformers import DataCollatorForSeq2Seq

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    output_dir="chess"

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        learning_rate=1e-3, # higher learning rate
        num_train_epochs=2,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="no",
        report_to="wandb",
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
	model=model,
        args=training_args,
	data_collator=data_collator,
	train_dataset=tokenized_dataset["train"],
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    trainer.train()

    # Save our LoRA model & tokenizer results
    peft_model_id="results"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
    # if you want to save the base model to call
    # trainer.model.base_model.save_pretrained(peft_model_id)

