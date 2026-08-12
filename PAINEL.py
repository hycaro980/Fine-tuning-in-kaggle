#==================================
#criado by HYC[]
#==================================

import os
import sys
import gc
import subprocess
import importlib


# ------------------------------------------------------------
# Corrigir torchao incompatível com PEFT
# ------------------------------------------------------------
def corrigir_torchao():
    print("Verificando torchao...")

    try:
        import importlib.metadata as metadata
        versao = metadata.version("torchao")
        print(f"torchao instalado: {versao}")

        partes = versao.split(".")
        major = int(partes[0]) if partes[0].isdigit() else 0
        minor = int(partes[1]) if len(partes) > 1 and partes[1].isdigit() else 0

        if major > 0 or minor >= 16:
            print("torchao está compatível.")
            return

        print("torchao incompatível detectado. Removendo...")

        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
            check=False
        )

        for nome in list(sys.modules.keys()):
            if nome == "torchao" or nome.startswith("torchao."):
                del sys.modules[nome]

        importlib.invalidate_caches()
        print("torchao removido.")

    except Exception:
        print("torchao não está instalado ou não foi detectado.")


corrigir_torchao()


# ------------------------------------------------------------
# Instalar dependências automaticamente, se necessário
# ------------------------------------------------------------
def instalar(pacotes):
    for pacote in pacotes:
        nome_import = pacote.replace("-", "_")
        try:
            __import__(nome_import)
        except ImportError:
            print(f"Instalando {pacote}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pacote])


instalar([
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "huggingface_hub"
])

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from huggingface_hub import login, hf_hub_download


# ------------------------------------------------------------
# Configurações gerais
# ------------------------------------------------------------
if os.path.exists("/kaggle/working"):
    OUTPUT_DIR = "/kaggle/working/SYNASTRIA-FINE-TUNING"
else:
    OUTPUT_DIR = "./SYNASTRIA-FINE-TUNING"

os.makedirs(OUTPUT_DIR, exist_ok=True)

STATE = {
    "model": None,
    "tokenizer": None,
    "base_model_name": None,
    "modo": None,          # "lora" ou "full"
    "adapter_path": None,
}


# ------------------------------------------------------------
# Funções auxiliares
# ------------------------------------------------------------
def limpar_memoria():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mostrar_gpus():
    print()
    if torch.cuda.is_available():
        print(f"GPUs detectadas: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("Nenhuma GPU detectada.")
        print("No Kaggle, ative GPU T4 x2 em Settings.")


def banner():
    print("=" * 62)
    print("SYNASTRIA-FINE-TUNING".center(62))
    print("Painel de Fine-Tuning - Kaggle GPU T4 x2".center(62))
    print("Sem Unsloth e sem bitsandbytes".center(62))
    print("=" * 62)


def perguntar(texto, padrao=None):
    while True:
        resposta = input(texto).strip()
        if resposta == "" and padrao is not None:
            return str(padrao)
        if resposta != "":
            return resposta
        print("Valor inválido. Tente novamente.")


def perguntar_int(texto, padrao):
    while True:
        try:
            return int(perguntar(texto, padrao))
        except ValueError:
            print("Digite um número inteiro.")


def perguntar_float(texto, padrao):
    while True:
        try:
            return float(perguntar(texto, padrao))
        except ValueError:
            print("Digite um número válido.")


def perguntar_opcao(texto, opcoes):
    while True:
        resposta = perguntar(texto)
        if resposta in opcoes:
            return resposta
        print(f"Opção inválida. Use uma destas: {', '.join(opcoes)}")


# ------------------------------------------------------------
# Carregar modelo e tokenizer
# ------------------------------------------------------------
def carregar_modelo_base(nome_modelo, device_map="auto"):
    print(f"\nCarregando modelo: {nome_modelo}")

    tokenizer = AutoTokenizer.from_pretrained(nome_modelo)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    modelo = AutoModelForCausalLM.from_pretrained(
        nome_modelo,
        torch_dtype=torch.float16,
        device_map=device_map
    )

    modelo.config.use_cache = False

    return modelo, tokenizer


# ------------------------------------------------------------
# Escolher target modules para LoRA
# ------------------------------------------------------------
def escolher_target_modules(nome_modelo):
    nome = nome_modelo.lower()

    if "gpt2" in nome:
        return ["c_attn"]

    if "opt" in nome:
        return ["q_proj", "v_proj"]

    if "phi" in nome:
        return ["q_proj", "k_proj", "v_proj", "dense"]

    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def aplicar_lora(modelo, nome_modelo):
    try:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            bias="none"
        )

        modelo = get_peft_model(modelo, lora_config)
        return modelo

    except Exception:
        target_modules = escolher_target_modules(nome_modelo)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none"
        )

        modelo = get_peft_model(modelo, lora_config)
        return modelo


# ------------------------------------------------------------
# Dataset (CORRIGIDO PARA ERRO DE CASTERROR)
# ------------------------------------------------------------
def carregar_dataset():
    print("\n--- DATASET ---")
    print("1) Dataset completo do Hugging Face (com 1 única configuração)")
    print("2) Arquivo local (JSON/JSONL/CSV)")
    print("3) Arquivo específico de um repositório do Hugging Face")
    print("   (Use esta opção se o repo tiver vários CSVs com colunas diferentes)")

    tipo = perguntar_opcao("Escolha uma opção: ", ["1", "2", "3"])
    ds = None

    if tipo == "1":
        nome = perguntar(
            "Nome do dataset no Hugging Face (ex: tatsu-lab/alpaca): ",
            "tatsu-lab/alpaca"
        )
        config = perguntar(
            "Nome da configuração/subset (deixe em branco se não tiver): ",
            ""
        )
        try:
            if config:
                ds = load_dataset(nome, name=config, split="train", trust_remote_code=True)
            else:
                ds = load_dataset(nome, split="train", trust_remote_code=True)
        except Exception as e:
            print(f"\nErro ao carregar o dataset completo: {type(e).__name__}")
            print("Isso geralmente acontece quando o repositório tem vários arquivos CSV com colunas diferentes.")
            print("Por favor, use a Opção 3 deste menu para escolher um arquivo específico.")
            return carregar_dataset() # Chama de novo

    elif tipo == "2":
        caminho = perguntar("Caminho do arquivo (ex: /kaggle/working/meu.jsonl): ")
        if caminho.endswith(".csv"):
            ds = load_dataset("csv", data_files=caminho, split="train")
        else:
            ds = load_dataset("json", data_files=caminho, split="train")

    elif tipo == "3":
        repo_id = perguntar("ID do repositório (ex: Skskskd/Test-heretic): ")
        arquivo = perguntar("Nome exato do arquivo no repositório (ex: dataset_psicologo_60exemplos.csv): ")
        
        print(f"Baixando {arquivo} de {repo_id}...")
        try:
            caminho_local = hf_hub_download(
                repo_id=repo_id,
                filename=arquivo,
                repo_type="dataset"
            )
        except Exception as e:
            print(f"Erro ao baixar o arquivo: {e}")
            return carregar_dataset()

        if caminho_local.endswith(".csv"):
            ds = load_dataset("csv", data_files=caminho_local, split="train")
        else:
            ds = load_dataset("json", data_files=caminho_local, split="train")

    print("\nColunas disponíveis:", ds.column_names)

    print("\nFormato do dataset:")
    print("1) Coluna única de texto")
    print("2) Instruction / Input / Output")

    formato = perguntar_opcao("Escolha uma opção: ", ["1", "2"])

    if formato == "1":
        coluna = perguntar("Nome da coluna de texto: ", "text")

        if coluna not in ds.column_names:
            print(f"Coluna '{coluna}' não encontrada.")
            print(f"Usando a primeira coluna disponível: {ds.column_names[0]}")
            coluna = ds.column_names[0]

        ds = ds.rename_column(coluna, "text")

    else:
        instr = perguntar("Coluna de instrução: ", "instruction")
        inp = perguntar("Coluna de entrada (deixe vazio se não existir): ", "")
        out = perguntar("Coluna de resposta: ", "output")

        def montar_texto(ex):
            instrucao = ex.get(instr, "")
            entrada = ex.get(inp, "") if inp else ""
            saida = ex.get(out, "")

            if entrada:
                texto = f"### Instrução:\n{instrucao}\n### Entrada:\n{entrada}\n### Resposta:\n{saida}"
            else:
                texto = f"### Instrução:\n{instrucao}\n### Resposta:\n{saida}"

            return {"text": texto}

        ds = ds.map(montar_texto)

    return ds


# ------------------------------------------------------------
# Fine-tuning
# ------------------------------------------------------------
def fazer_finetuning():
    limpar_memoria()

    print("\n--- FINE-TUNING ---")

    modelos_sugeridos = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "gpt2",
        "facebook/opt-125m",
    ]

    print("\nModelos sugeridos para T4 sem quantização:")
    for i, m in enumerate(modelos_sugeridos, 1):
        print(f"{i}) {m}")
    print("0) Digitar outro modelo do Hugging Face")

    escolha = perguntar("Escolha o modelo: ", "1")

    if escolha == "0":
        nome_modelo = perguntar("Nome do modelo no Hugging Face: ")
    elif escolha.isdigit() and 1 <= int(escolha) <= len(modelos_sugeridos):
        nome_modelo = modelos_sugeridos[int(escolha) - 1]
    else:
        nome_modelo = escolha

    print("\nModo de treino:")
    print("1) LoRA (recomendado para T4)")
    print("2) Full fine-tuning (apenas modelos muito pequenos)")

    modo = perguntar_opcao("Escolha: ", ["1", "2"])

    if modo == "1":
        device_map = "auto"
        STATE["modo"] = "lora"
    else:
        device_map = {"": 0} if torch.cuda.is_available() else None
        STATE["modo"] = "full"

    modelo, tokenizer = carregar_modelo_base(nome_modelo, device_map=device_map)

    if modo == "1":
        print("\nAplicando LoRA...")
        modelo = aplicar_lora(modelo, nome_modelo)
        modelo.print_trainable_parameters()

    ds = carregar_dataset()

    max_len = perguntar_int("Tamanho máximo de tokens (padrão 512): ", 512)
    epochs = perguntar_int("Número de épocas (padrão 1): ", 1)
    batch = perguntar_int("Batch size por GPU (padrão 2): ", 2)
    grad_acc = perguntar_int("Gradient accumulation steps (padrão 4): ", 4)

    lr_padrao = 2e-4 if modo == "1" else 1e-5
    lr = perguntar_float(f"Learning rate (padrão {lr_padrao}): ", lr_padrao)

    def tokenizar(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len,
            padding="max_length"
        )

    print("\nTokenizando dataset...")
    ds = ds.map(
        tokenizar,
        batched=True,
        remove_columns=ds.column_names
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    output_dir = os.path.join(OUTPUT_DIR, "checkpoints")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
        logging_dir=os.path.join(OUTPUT_DIR, "logs")
    )

    trainer = Trainer(
        model=modelo,
        args=args,
        train_dataset=ds,
        data_collator=collator
    )

    print("\nIniciando treino...")
    trainer.train()

    save_dir = os.path.join(OUTPUT_DIR, "modelo_final")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nSalvando modelo em: {save_dir}")

    modelo.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    STATE["model"] = modelo
    STATE["tokenizer"] = tokenizer
    STATE["base_model_name"] = nome_modelo

    if STATE["modo"] == "lora":
        STATE["adapter_path"] = save_dir

    print("\nTreino concluído com sucesso!")


# ------------------------------------------------------------
# Gerar texto
# ------------------------------------------------------------
def gerar_texto(model, tokenizer, prompt, max_new_tokens=128, temperature=0.7):
    model.eval()

    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        saida = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(saida[0], skip_special_tokens=True)


# ------------------------------------------------------------
# Testar modelo
# ------------------------------------------------------------
def testar_modelo(usar_ultimo=False):
    print("\n--- TESTE DE MODELO ---")

    if usar_ultimo and STATE["model"] is not None:
        model = STATE["model"]
        tokenizer = STATE["tokenizer"]
        print(f"Usando modelo já carregado: {STATE['base_model_name']}")

    else:
        print("1) Testar último modelo treinado")
        print("2) Testar modelo do Hugging Face")

        opcao = perguntar_opcao("Escolha: ", ["1", "2"])

        if opcao == "1":
            if STATE["model"] is None:
                print("Nenhum modelo treinado nesta sessão.")
                return

            model = STATE["model"]
            tokenizer = STATE["tokenizer"]

        else:
            nome = perguntar("Nome do modelo no Hugging Face: ")
            model, tokenizer = carregar_modelo_base(nome, device_map="auto")

    print("\nModo de teste interativo.")
    print("Digite 'sair' para encerrar o teste.")

    while True:
        prompt = perguntar("\nPrompt: ")

        if prompt.lower() == "sair":
            break

        print("\nGerando resposta...")
        resposta = gerar_texto(model, tokenizer, prompt)

        print("\nResposta:")
        print(resposta)


# ------------------------------------------------------------
# Postar no Hugging Face
# ------------------------------------------------------------
def postar_huggingface():
    print("\n--- POSTAR NO HUGGING FACE ---")

    if STATE["model"] is None:
        print("Nenhum modelo treinado para postar.")
        return

    print("\nVocê precisa de um token do Hugging Face.")
    print("Crie em: https://huggingface.co/settings/tokens")

    token = perguntar("Token de acesso do Hugging Face: ")

    login(token=token, add_to_git_credential=False)

    repo_id = perguntar("Nome do repositório (ex: seu-usuario/synastria-modelo): ")

    modelo = STATE["model"]
    tokenizer = STATE["tokenizer"]

    print("\nO que deseja postar?")
    print("1) Adapter LoRA (leve)")
    print("2) Modelo completo merged (mais pesado)")

    opcao = perguntar_opcao("Escolha: ", ["1", "2"])

    if opcao == "1":
        print("\nEnviando adapter LoRA...")
        modelo.push_to_hub(repo_id, token=token)
        tokenizer.push_to_hub(repo_id, token=token)

        print(f"\nAdapter LoRA enviado para: https://huggingface.co/{repo_id}")

    else:
        if isinstance(modelo, PeftModel):
            print("\nMesclando LoRA ao modelo base...")
            modelo = modelo.merge_and_unload()

        print("\nEnviando modelo completo...")
        modelo.push_to_hub(repo_id, token=token)
        tokenizer.push_to_hub(repo_id, token=token)

        print(f"\nModelo completo enviado para: https://huggingface.co/{repo_id}")


# ------------------------------------------------------------
# Menu pós-treino
# ------------------------------------------------------------
def menu_pos_treino():
    while True:
        print("\n--- O QUE DESEJA FAZER AGORA? ---")
        print("1) Testar modelo")
        print("2) Postar no Hugging Face")
        print("3) Treinar novamente")
        print("4) Voltar ao menu principal")

        opcao = perguntar_opcao("Escolha: ", ["1", "2", "3", "4"])

        if opcao == "1":
            testar_modelo(usar_ultimo=True)

        elif opcao == "2":
            postar_huggingface()

        elif opcao == "3":
            fazer_finetuning()

        else:
            break


# ------------------------------------------------------------
# Menu principal
# ------------------------------------------------------------
def menu_principal():
    banner()
    mostrar_gpus()

    while True:
        print("\n--- MENU PRINCIPAL ---")
        print("1) Fazer fine-tuning")
        print("2) Testar modelos")
        print("3) Sair")

        opcao = perguntar_opcao("Escolha: ", ["1", "2", "3"])

        if opcao == "1":
            fazer_finetuning()
            menu_pos_treino()

        elif opcao == "2":
            testar_modelo(usar_ultimo=False)

        else:
            print("\nSaindo do SYNASTRIA-FINE-TUNING...")
            break


# ------------------------------------------------------------
# Execução
# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        menu_principal()
    except KeyboardInterrupt:
        print("\nEncerrado pelo usuário.")
    except Exception as e:
        print(f"\nErro: {e}")
        raise
