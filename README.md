# Implementation Code for A Study of Implicit Ranking Unfairness in Large Language Models of EMNLP 2024 findings

## If you any questions, please mail to 
```
Xu Chen: xc_chen@ruc.edu.cn
```


We will illustrate the experiment from next steps:
## 1. Generate responses from LLMs
### For run ChatGPT:
```bash
python run_chatgpt.py [arg1] [arg2] ... [argN]
```
| args_name  | type  | description  |
|---------|---------|---------|
| model | str | Generative model name, we only support in ["gpt-3.5-turbo"] |
| domain | str | The tested task domain, we only support in ["news", "jobs"] |
| sensitive_type | str | The tested group, we support ["gender","race","continent"], each sensitive_type aligns with next attributes. |
| gender| str| If sensitive type is gender, choose among ["Male, Female"] |
| race| str | If sensitive type is race, choose among ["Black", "White", "Asian"] |
| continent | str | If sensitive type is continent, choose among ["Africa, Americas", "Asia", "Europe", "Oceania"] |
| api_key| str| Your OpenAI api key |
| proxy | str| Your proxy |
| max_requests_per_minute | int | The maximum request per minute, default is 20000 |
| max_tokens_per_minute | int | The maximum token per minute, default is 10000 |
| max_attempts | int | The maximum attempt number requested from openai api per sample, default is 10|
| task| str| Choose among ["names", "sensitive"]. Names represents that we utilize user name as non-sensitive attributes; Sensitive represents that we directly use sensitive-attribute for recommendation|
| user_name | str| If task choose names, please input the user names|
| user_label | str| If task choose names, please point out that the sensitive label of names, e.g. "Bob" is Male name, the label should be the index of ["Male, Female"]: 0 |

The example results are: request.example and out.example

### According to different LLMs' format, parse the format to csv

| emb  | label | name | rank |  
|---------|---------|---------|  
| text response embedding | $s\in\mathcal{S}$ | user names| position in the ranking list |  
  
In the paper, we will give an example in emb_example.csv


## 2. Choose the proper user names

run bash
```
python filter_attribute.py
```
Then you will get the score of each names: {'Alice': 6.37, 'Bob': 5.99}.
Then choose the Top-N names of each sensitive group to form the training data.

## 3. Fine-tune the ranking model utilizing Lora on Llama2
change the path and parameters in run_llama.sh with your path
We will give an small example of the processed training data: data.json
run bash
```
sh run_llama.sh
```

## For citation, please ref

```
@inproceedings{xu-etal-2024-multi,
    title = "A Study of Implicit Ranking Unfairness in Large Language Models",
    author = "Xu, Chen  and
      Wang, Wenjie  and
      Li, Yuxin  and
      Pang, Liang  and
      Xu, Jun and Tat-Seng Chua",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = Nov,
    year = "2024",
    address = "Miami, Florida",
    publisher = "Association for Computational Linguistics"}
```

