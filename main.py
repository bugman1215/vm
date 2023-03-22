import torch
import transformers

# 以 utf-8 的编码格式打开指定文件

f = open("west.txt", encoding="utf-8")

# 输出读取到的数据

# print(f.read())

# 关闭文件


from transformers import (
    EncoderDecoderModel,
    AutoTokenizer
)

PRETRAINED = "raynardj/wenyanwen-ancient-translate-to-modern"
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
model = EncoderDecoderModel.from_pretrained(PRETRAINED)


def inference(text):
    tk_kwargs = dict(
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors='pt')

    inputs = tokenizer([text, ], **tk_kwargs)
    with torch.no_grad():
        return tokenizer.batch_decode(
            model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                num_beams=3,
                max_length=256,
                bos_token_id=101,
                eos_token_id=tokenizer.sep_token_id,
                pad_token_id=tokenizer.pad_token_id,
            ), skip_special_tokens=True)


Note = open('westtranslate.txt', mode='w')

i = 0
texts = f.read()
ls = texts.split('。')
length = len(ls)
while i < length:
    Note.writelines(inference(ls[i]))
    Note.write('\n')
    # print(inference(ls[i]))
    i = i + 1
f.close()

