# 🌌 SYNASTRIA-FINE-TUNING

> **Desenvolvido por Synastria Networks**  
> Painel de Fine-Tuning Interativo | Otimizado para Kaggle GPU T4 x2 | Sem Unsloth & Sem bitsandbytes

[![Discord](https://img.shields.io/badge/Discord-Synastria%20Networks-7289da?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/wbwj9bBBGa)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=for-the-badge&logo=pytorch)]()
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)]()

## 📋 Sobre o Projeto

O **SYNASTRIA-FINE-TUNING** é uma ferramenta CLI (Command Line Interface) completa e autônoma para ajuste fino (fine-tuning) de modelos de linguagem (LLMs). Projetado especificamente para rodar em ambientes com GPUs limitadas como o **Kaggle (T4 x2)**, este script elimina dependências problemáticas como `bitsandbytes` e `unsloth`, garantindo estabilidade através de correções automáticas de ambiente e gerenciamento inteligente de memória.

### ✨ Funcionalidades Principais

- 🛠️ **Auto-Correção de Ambiente:** Detecta e remove versões incompatíveis do `torchao` automaticamente.
- 📦 **Instalação Automática:** Verifica e instala todas as dependências necessárias (`transformers`, `peft`, `accelerate`, etc.) ao iniciar.
- 🎯 **Suporte a LoRA e Full Fine-Tuning:** Configuração simplificada com fallback inteligente de `target_modules`.
- 📊 **Carregamento Flexível de Datasets:** Suporte a datasets do Hugging Face, arquivos locais (JSON/CSV) ou download direto de repositórios HF.
- 💬 **Teste Interativo Integrado:** Chat em tempo real no terminal logo após o treino.
- ☁️ **Push to Hub Nativo:** Envie adapters LoRA ou modelos completos mesclados diretamente para o Hugging Face.
- 🧹 **Gestão de Memória:** Limpeza automática de cache CUDA e GC entre operações.

---

## 🚀 Como Usar

### Pré-requisitos
- Python 3.10+
- GPU NVIDIA recomendada (Script otimizado para Kaggle T4 x2)
- Token de acesso do Hugging Face (para upload de modelos)

### Execução

Basta rodar o script principal. As dependências serão instaladas automaticamente na primeira execução:

PAINEL.py
