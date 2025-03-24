# Book Helper

```
api_module.py is the main module that handles groq api and runware image generation model. 
api_module depends on utils.py,audio_module.py,prompts.py and logger_module.py

api_module requires GROQ_API in .env file 
utils requires HF_API in .env file

make sure the HF account tht HF_API belongs to has access to "meta-llama/Llama-3.1-8B-Instruct" model 
you can go to the huggingface repo of "meta-llama/Llama-3.1-8B-Instruct" to get access 
```

**Note - HG_API and Google_api is private (not included in repo)**

`.env` -> `HF_API=hf_F*********************************`
