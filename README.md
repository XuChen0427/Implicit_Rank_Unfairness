### Implementation Code for "Are LLMs Pretending To Be Fair? Goodhart's Law in Evaluating Discrimination of Large Language Models"

We will illustrate the experiment from next steps:
## 1. Generate responses from LLMs
### For run ChatGPT:
```bash
python run_chatgpt.py [arg1] [arg2] ... [argN]
```
| args_name  | type  | description  |
|---------|---------|---------|
| model | str | Generative model name, we only support in ["gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613"] |
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
| emb  | label | rank |
|---------|---------|---------|
| text response embedding from | $s\in\mathcal{S}$ | position in the ranking list |


## 2. Train IV to get weight for each responses
```bash
python train_IV.py
```
Get the weight npy file for testing
