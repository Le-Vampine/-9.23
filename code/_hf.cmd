@echo off
set PYTHONUTF8=1
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=E:\hf_cache
cd /d "%~dp0"
if exist _hf.log del _hf.log
python -m pip install transformers huggingface_hub safetensors -i https://pypi.tuna.tsinghua.edu.cn/simple >> _hf.log 2>&1
echo --- pip done --- >> _hf.log
python -u -c "from transformers import AutoTokenizer, AutoModel; n='bert-base-uncased'; tk=AutoTokenizer.from_pretrained(n); mo=AutoModel.from_pretrained(n); print('OK params=%.1fM' % (sum(p.numel() for p in mo.parameters())/1e6)); print('vocab', tk.vocab_size)" >> _hf.log 2>&1
echo FINISHED >> _hf.log
