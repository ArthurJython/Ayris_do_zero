# AYRIS

Projeto experimental de modelo de linguagem autoregressivo, treinado do zero, com pesos inicialmente aleatórios. O desenvolvimento começou com pré-treinamento em português usando livros e documentos. A etapa atual adiciona um pipeline separado de fine-tuning supervisionado para conversação.

O objetivo de longo prazo é construir um sistema modular: um núcleo conversacional com personalidade própria, ferramentas externas coordenadas por um orquestrador e um componente experimental `think` baseado em NAT (*Non-Autoregressive Transformer*). A arquitetura e a integração do NAT ainda não foram definidas ou implementadas.

> **Estado atual:** arquitetura Transformer implementada; tokenizer BPE de 4.096 tokens; pré-treinamento literário das fases 1–4 concluído; checkpoint da fase 4 disponível; dataset conversacional inicial de 24 diálogos preparado e validado; `ChatConversationDataset` e loss mascarada implementados; teste de integração de uma atualização executado. O fine-tuning conversacional completo ainda precisa ser executado e avaliado.

---

## 1. Ambiente e hardware

- Sistema: Pop!_OS / Linux.
- CPU: Intel Core i5-6500.
- RAM: 8 GB.
- GPU/CUDA: não utilizada; treinamento em CPU.
- Precisão: FP32.
- Ambiente virtual: `.venv/` na raiz do projeto.
- Caminho do projeto: `~/Documentos/AYRIS/AYRIS`.
- Python registrado: 3.13.5.
- PyTorch registrado: 2.15.0+cpu.
- Confirme as versões efetivas com `python --version` e `python -c "import torch; print(torch.__version__)"`; não atualizar dependências sem verificar compatibilidade.

Entrar na raiz e ativar o ambiente:

```bash
cd ~/Documentos/AYRIS/AYRIS
source .venv/bin/activate
```

Dependências principais: `torch`, `numpy`, `pyarrow` e `tokenizers`. O pipeline de dados original também utiliza arquivos Parquet. Consulte `requirements.txt` antes de instalar ou atualizar pacotes.

---

## 2. Arquitetura atual do modelo

AYRIS é um modelo decoder-only autoregressivo, com atenção causal.

| Componente | Configuração |
|---|---:|
| Vocabulário BPE | 4.096 tokens |
| Token embedding | 256 dimensões |
| Transformer blocks | 11 |
| `d_model` | 256 |
| Cabeças de atenção | 4 |
| Dimensão por cabeça | 64 |
| MLP | 1.024 |
| Posicionamento | RoPE, base 10.000 |
| Atenção | Causal Self-Attention |
| Normalização | Pre-LayerNorm e LayerNorm final |
| Ativação MLP | GELU |
| Contexto máximo | 512 tokens |
| Precisão | FP32 |
| Embedding/LM head | Weight tying |
| Parâmetros únicos | 9.736.448 |

Tokens especiais do tokenizer:

| Token | ID |
|---|---:|
| `<PAD>` | 0 |
| `<UNK>` | 1 |
| `<BOS>` | 2 |
| `<EOS>` | 3 |
| `<SYS>` | 4 |
| `<USR>` | 5 |
| `<AST>` | 6 |

O modelo-base continua sendo autoregressivo. RoPE fornece informação posicional à atenção, mas não ensina conversação ou raciocínio por si só. Essas capacidades dependem dos dados, do objetivo de treinamento e da arquitetura completa do sistema.

O projeto ainda não possui o componente NAT implementado. A ideia futura é investigar um módulo `think` separado; não se deve tratar essa proposta como uma capacidade disponível no modelo atual.

---

## 3. Pré-treinamento literário

### 3.1 Corpus

Arquivos de origem do corpus literário:

```text
data_0.parquet
data/processed/data_0_clean.parquet
data/splits/
├── train.parquet
├── validation.parquet
├── test.parquet
└── split_manifest.json
```

O split foi feito por documento (`filename`), sem sobreposição entre treino, validação e teste, conforme auditorias anteriores. A suíte de testes original registrou 58/58 testes aprovados antes da adição do pipeline conversacional; execute novamente a suíte após alterações recentes para confirmar o estado atual.

| Divisão | Registros | Documentos | Tokens tokenizados |
|---|---:|---:|---:|
| Treino | 26.226 | 1.213 | 98.501.741 |
| Validação | 513 | 26 | 1.940.433 |
| Teste | 661 | 27 | 2.476.215 |
| **Total** | **27.400** | **1.266** | **102.918.389** |

O corpus é predominantemente composto por prosa de livros e documentos, incluindo textos históricos e formais. Não é um corpus conversacional. Foram observados problemas de extração de texto, como espaços ausentes e palavras concatenadas. Não reconstruir os fragmentos indiscriminadamente com `" ".join()` nem concatenar registros sem respeitar os limites originais.

Política de preparação do corpus literário:

- Tokenizar cada fragmento/registro separadamente.
- Não concatenar fragmentos artificialmente.
- Inserir `<BOS>` apenas no primeiro fragmento de cada documento.
- Inserir `<EOS>` apenas no último fragmento de cada documento.
- Não criar janelas que atravessem limites de fragmentos.
- Não utilizar `<PAD>` no pipeline atual.

Arquivos tokenizados:

```text
data/tokenized/
├── train.bin
├── train.index.jsonl
├── validation.bin
├── validation.index.jsonl
├── test.bin
├── test.index.jsonl
└── manifest.json
```

Os IDs são armazenados em `uint16` little-endian. O `TokenWindowDataset` utiliza `numpy.memmap` e cria janelas de 512 tokens, com entrada e alvo deslocados em um token. Os dados de treino registrados anteriormente contêm 205.482 janelas; a validação contém 4.054 janelas.

### 3.2 Fases e resultados registrados

As fases literárias são etapas de treinamento do mesmo modelo. Cada nova fase normalmente inicializa os pesos a partir do melhor checkpoint da fase anterior e reinicializa AdamW e o scheduler. Isso é diferente de retomar uma execução interrompida.

| Fase | Atualizações | Batch | Resultado registrado |
|---|---:|---:|---|
| Piloto | 20 | 1 | Experimento inicial; serviu como ponto de partida para a fase 2. |
| Fase 2 | 1.000 | 1 | Validation Loss final: 6,16481; PPL: 475,71. |
| Fase 3 | 5.000 | 1 | Validation Loss: 5,380750; PPL: 217,1851; Test Loss: 5,240537; Test PPL: 188,7714. |
| Fase 4 | 100.000 | 3 | Validation Loss: 3,7197099; PPL: 41,2524. |

Detalhes registrados da fase 4:

- Inicializada a partir do melhor checkpoint da fase 3.
- 100.000 atualizações na fase 4.
- Batch size 3 e contexto de 512 tokens: 1.536 tokens-alvo por atualização.
- Aproximadamente 153,6 milhões de posições de tokens processadas durante a fase.
- Training Loss do último intervalo: aproximadamente 3,3958919.
- Validation Loss final: aproximadamente 3,7197099.
- Diferença final entre validação e treino: aproximadamente 0,323818.
- A validação periódica utilizou 128 janelas fixas (65.536 tokens), não o conjunto de validação inteiro.
- Tempo acumulado registrado: aproximadamente 25h48m55s.
- O melhor resultado de validação ocorreu no passo 100.000.

A validação periódica por amostragem fixa permite comparar checkpoints sob o mesmo protocolo, mas não substitui uma avaliação final em todas as janelas de validação. O conjunto `test` deve permanecer reservado para avaliação final, sem ser usado para escolher checkpoints ou ajustar hiperparâmetros.

A geração textual da fase 4 mostrou frases mais longas e formas reconhecíveis de português, mas ainda apresentou repetição, vagueza semântica e progressão fraca de ideias. Isso não comprova compreensão factual nem capacidade conversacional confiável.

---

## 4. Como usar o treinador literário

O treinador original permanece em:

```text
src/training/trainer.py
```

Ele usa o dataset literário em `data/tokenized/`, com janelas de 512 tokens, e calcula Cross Entropy nos alvos de todos os tokens da janela.

### 4.1 Iniciar uma nova fase literária a partir de um checkpoint

Exemplo de uma nova fase de 5.000 passos, inicializada a partir do melhor checkpoint da fase 4:

```bash
python -m src.training.trainer \
  --steps 5000 \
  --batch-size 1 \
  --run-name fase_05_literaria_5000 \
  --threads 2 \
  --learning-rate 0.0001 \
  --warmup-steps 1000 \
  --init-from checkpoints/training/fase_04_100k/ayris_best.pt
```

Este é um exemplo de comando, não uma recomendação para executar automaticamente. Escolha um `run-name` novo e defina os hiperparâmetros de acordo com o experimento. O código do treinador contém defaults próprios; informe explicitamente os valores que deseja usar.

`--init-from` carrega os pesos do checkpoint indicado, mas cria um novo AdamW e um novo scheduler. O contador da nova execução começa em zero.

### 4.2 Retomar uma execução literária interrompida

Use `--resume` com o mesmo `run-name` e a mesma configuração da execução original:

```bash
python -m src.training.trainer \
  --steps 5000 \
  --batch-size 1 \
  --run-name fase_05_literaria_5000 \
  --threads 2 \
  --learning-rate 0.0001 \
  --warmup-steps 1000 \
  --resume
```

Para `--resume`, mantenha iguais os valores que fazem parte da configuração salva, incluindo total de passos, batch size, learning rate, warmup e intervalos. O treinador também verifica a impressão digital dos dados e do tokenizer.

Não use `--init-from` para retomar uma execução interrompida:

- `--init-from`: inicia uma nova fase com pesos existentes e otimizador/scheduler novos.
- `--resume`: recupera modelo, otimizador, scheduler, sampler e estados aleatórios da mesma execução.
- `--from-scratch`: inicializa pesos aleatórios; usar somente quando a intenção for começar um pré-treinamento do zero.

### 4.3 Hiperparâmetros padrão do treinador literário

Os valores definidos no código atual informado para `trainer.py` são:

| Parâmetro | Default |
|---|---:|
| Passos | 100.000 |
| Batch size | 3 |
| Contexto | 512 |
| Learning rate máximo | `1e-4` |
| Warmup | 1.000 passos |
| AdamW betas | `(0.9, 0.95)` |
| Adam epsilon | `1e-8` |
| Weight decay | `0.1` para matrizes e `0.0` para parâmetros 1D |
| Gradient clipping | `1.0` |
| Avaliação | 128 janelas fixas |
| Intervalo de avaliação | 5.000 passos |
| Salvamento de `latest` | 1.000 passos |
| Log | 250 passos |

A CLI permite alterar diversos desses parâmetros. Consulte `python -m src.training.trainer --help` para confirmar os argumentos disponíveis no arquivo local.

### 4.4 Saída do treinador literário

Cada execução cria uma pasta em `checkpoints/training/<run-name>/`, normalmente contendo:

```text
checkpoints/training/<run-name>/
├── ayris_best.pt
├── ayris_latest.pt
├── metrics.jsonl
└── stages/                 # snapshots de pesos, quando habilitados
```

- `ayris_best.pt`: checkpoint com a menor Validation Loss observada.
- `ayris_latest.pt`: estado mais recente, incluindo estado do otimizador e sampler, para retomar.
- `metrics.jsonl`: histórico de treinamento e validação.
- `stages/`: snapshots históricos de pesos; não substituem o checkpoint completo de retomada.

---

## 5. Fine-tuning conversacional

O fine-tuning conversacional é separado do pré-treinamento literário. Ele utiliza os pesos de uma fase literária concluída e treina a AYRIS com exemplos no formato de diálogo.

O objetivo desta etapa não é ensinar novamente toda a língua portuguesa. É adaptar o modelo para utilizar a base linguística em interações com mensagens de usuário e respostas da AYRIS.

### 5.1 Estrutura de arquivos

```text
data/chat/
├── conversations.txt             # conversas aprovadas para treinamento
├── raw/
│   └── conversations_raw.txt     # material experimental ainda não aprovado
└── tokenized/
    ├── train.bin
    ├── train.mask.bin
    ├── train.index.jsonl
    ├── validation.bin
    ├── validation.mask.bin
    ├── validation.index.jsonl
    ├── test.bin
    ├── test.mask.bin
    ├── test.index.jsonl
    └── manifest.json
```

O arquivo `conversations.txt` contém o conjunto aprovado. O arquivo em `raw/` serve para guardar exemplos em elaboração; ele não é lido automaticamente pelo preparador.

Formato esperado de cada conversa:

```text
<BOS><SYS>Instruções de personalidade da AYRIS.<USR>Mensagem do usuário.<AST>Resposta da AYRIS.<USR>Próxima mensagem.<AST>Próxima resposta.<EOS>
```

Cada diálogo deve ser uma unidade completa. Uma conversa não deve ser dividida entre treino, validação e teste. Os marcadores especiais devem ser preservados exatamente.

### 5.2 Preparar ou atualizar o dataset conversacional

Script:

```text
src/data/prepare_chat_dataset.py
```

Execute a partir da raiz do projeto:

```bash
python -m src.data.prepare_chat_dataset
```

O script:

1. Lê `data/chat/conversations.txt`.
2. Verifica os blocos delimitados por `<BOS>` e `<EOS>`.
3. Confere a presença dos marcadores de sistema, usuário e assistente.
4. Carrega o tokenizer existente em `checkpoints/tokenizer/tokenizer.json`.
5. Confirma os IDs dos marcadores: BOS 2, EOS 3, SYS 4, USR 5 e AST 6.
6. Separa conversas completas em treino, validação e teste, com seed fixa 42.
7. Salva tokens em `uint16` little-endian e máscaras em `uint8`.
8. Salva índices JSONL com offset e comprimento de cada conversa.
9. Gera `manifest.json` com hashes para verificar a compatibilidade dos dados e do tokenizer.

Para o conjunto inicial de 24 conversas, a separação gerada foi:

| Divisão | Conversas | Tokens | Tokens marcados para loss |
|---|---:|---:|---:|
| Treino | 20 | 1.961 | 1.212 |
| Validação | 2 | 166 | 96 |
| Teste | 2 | 200 | 134 |
| **Total** | **24** | **2.327** | **1.442** |

Os totais de tokens de entrada do `ChatConversationDataset` são menores que os totais do arquivo porque cada exemplo retorna `input_ids = ids[:-1]` e `target_ids = ids[1:]`.

Sempre execute novamente o preparador depois de editar `conversations.txt`. Não edite manualmente os arquivos binários ou o manifesto.

### 5.3 Máscara de loss: o que o modelo aprende

Arquivos relevantes:

```text
src/data/chat_dataset.py
src/training/chat_step.py
```

`ChatConversationDataset` carrega uma conversa completa e retorna três tensores:

- `input_ids`: sequência de entrada para o modelo.
- `target_ids`: sequência deslocada em um token, que representa os próximos tokens.
- `loss_mask`: máscara booleana que seleciona os alvos usados no cálculo da loss.

A máscara atual seleciona o conteúdo das respostas da AYRIS e o `<EOS>` final. Ela não seleciona as instruções `<SYS>`, as mensagens `<USR>` nem o marcador `<AST>`.

**A máscara não remove contexto.** O modelo continua recebendo as instruções e todas as mensagens anteriores do diálogo. A máscara apenas define em quais posições a Cross Entropy é calculada.

Exemplo conceitual:

| Parte da sequência | Entra como contexto? | Participa da loss? |
|---|---|---|
| `<BOS>` | Sim | Não |
| `<SYS>` e instruções | Sim | Não |
| `<USR>` e mensagem do usuário | Sim | Não |
| `<AST>` | Sim | Não |
| Texto da resposta da AYRIS | Sim | Sim |
| `<EOS>` da conversa | Sim | Sim |

`train_chat_step` executa forward, Cross Entropy mascarada, backpropagation, clipping de gradiente e AdamW. O treinamento é supervisionado: o modelo recebe o histórico e aprende a prever os tokens da resposta.

### 5.4 Limites do dataset inicial

O primeiro conjunto tem apenas 20 conversas de treino, 2 de validação e 2 de teste. Ele é suficiente para verificar o funcionamento do pipeline e realizar um experimento curto, mas é pequeno demais para demonstrar uma capacidade conversacional generalizável.

Riscos principais:

- Memorização das frases dos exemplos.
- Repetição de expressões e estruturas presentes nos poucos diálogos.
- Validação instável por conter apenas duas conversas.
- Baixa diversidade de temas, intenções, tons e comprimentos.

A Validation Loss desse conjunto não deve ser interpretada como uma medida robusta da qualidade conversacional. A avaliação qualitativa com prompts fixos e exemplos não vistos no treino é necessária.

---

## 6. Como usar o treinador de chat

O novo treinador é separado do treinador literário:

```text
src/training/chat_trainer.py
```

Ele utiliza `data/chat/tokenized/` e não altera `data/tokenized/`. O treinador de chat mantém o modelo AYRIS atual, inicializa os pesos a partir de um checkpoint literário e cria um otimizador e scheduler próprios para o fine-tuning.

> **Estado experimental:** o dataset, a classe `ChatConversationDataset` e a atualização individual com loss mascarada passaram pelos testes descritos acima. O treinamento completo com `chat_trainer.py` ainda deve ser executado e inspecionado; não tratar o experimento como concluído antes de verificar seus logs e checkpoints locais.

### 6.1 Piloto de 40 atualizações

Comando de piloto proposto:

```bash
python -m src.training.chat_trainer \
  --steps 40 \
  --learning-rate 0.00001 \
  --warmup-steps 2 \
  --eval-interval 20 \
  --save-interval 10 \
  --threads 2 \
  --run-name chat_finetune_v1 \
  --init-from checkpoints/training/fase_04_100k/ayris_best.pt
```

Esse comando:

- Inicializa os pesos a partir de `fase_04_100k/ayris_best.pt`.
- Usa batch size fixo 1 (uma conversa por atualização).
- Utiliza learning rate máximo `1e-5` e warmup de 2 passos.
- Executa 40 atualizações, aproximadamente duas passagens pelo conjunto de 20 conversas de treino.
- Avalia as duas conversas do split de validação a cada 20 passos e no passo final.
- Salva checkpoints a cada 10 passos e ao finalizar.
- Utiliza duas threads de CPU.

O piloto é curto de propósito. Não aumente o número de passos automaticamente com base apenas na Validation Loss; primeiro compare as gerações antes e depois do fine-tuning.

### 6.2 Retomar o fine-tuning de chat

Use os mesmos argumentos de configuração e substitua `--init-from` por `--resume`:

```bash
python -m src.training.chat_trainer \
  --steps 40 \
  --learning-rate 0.00001 \
  --warmup-steps 2 \
  --eval-interval 20 \
  --save-interval 10 \
  --threads 2 \
  --run-name chat_finetune_v1 \
  --resume
```

O `--resume` exige que o diretório da execução exista e que a configuração seja idêntica à salva. Não reutilize o mesmo `run-name` para iniciar outro experimento independente.

### 6.3 Saída do fine-tuning conversacional

```text
checkpoints/training/chat_finetune_v1/
├── ayris_best.pt
├── ayris_latest.pt
└── metrics.jsonl
```

- `ayris_best.pt`: pesos e estados salvos quando a Validation Loss melhora.
- `ayris_latest.pt`: checkpoint mais recente, com estado do otimizador, scheduler e sampler para retomada.
- `metrics.jsonl`: eventos de treinamento, validação e interrupção.

O treinador de chat descrito atualmente usa o split de treino e o de validação. O split `test` fica reservado e não é utilizado automaticamente por esse script. Deve ser usado em uma avaliação final separada, sem influenciar a escolha do checkpoint.

### 6.4 Teste de integração já realizado

Foi executada uma atualização de `train_chat_step` usando um `AYRISModel` recém-inicializado aleatoriamente, sem carregar o checkpoint da fase 4. Resultado registrado:

- Loss: aproximadamente `10,0762`.
- Tokens de resposta selecionados: 39.
- Norma do gradiente antes do clipping: aproximadamente `46,81`.
- Learning rate: `1e-5`.

Esse teste confirmou que dataset, modelo e loss mascarada conseguem executar uma atualização. **Não é uma avaliação da qualidade da AYRIS treinada**, pois o modelo desse teste tinha pesos aleatórios e não foi inicializado com o checkpoint literário.

---

## 7. Comparação entre os dois treinadores

Os dois treinadores usam a mesma arquitetura neural, mas conjuntos de dados e objetivos de loss diferentes.

| Característica | `trainer.py` literário | `chat_trainer.py` conversacional |
|---|---|---|
| Dataset | `data/tokenized/` | `data/chat/tokenized/` |
| Unidade de amostra | Janela de 512 tokens | Conversa completa |
| Modelo | `AYRISModel` atual | O mesmo `AYRISModel` |
| Inicialização típica | Checkpoint literário anterior | Melhor checkpoint literário |
| Objetivo | Prever todos os tokens-alvo | Prever apenas respostas da AYRIS |
| Loss | Cross Entropy comum | Cross Entropy mascarada |
| Contexto de usuário | Não é o formato central do corpus | Sim; mensagens anteriores são contexto |
| Otimizador | AdamW | AdamW independente |
| Checkpoints | Diretório próprio da fase | Diretório próprio do fine-tuning |

O fine-tuning de chat não cria uma segunda rede neural. Ele atualiza os pesos do mesmo modelo, partindo de um checkpoint literário. Para preservar a base anterior, nunca sobrescreva os checkpoints das fases literárias.

---

## 8. Avaliação e geração

O script de inferência conhecido é:

```text
src/inference/generate.py
```

Exemplo para testar o melhor checkpoint literário da fase 4:

```bash
python -m src.inference.generate \
  --checkpoint checkpoints/training/fase_04_100k/ayris_best.pt \
  --prompt "A capital do Brasil é"
```

Para testar o checkpoint de chat, use o mesmo comando apontando para:

```text
checkpoints/training/chat_finetune_v1/ayris_best.pt
```

Exemplo:

```bash
python -m src.inference.generate \
  --checkpoint checkpoints/training/chat_finetune_v1/ayris_best.pt \
  --prompt "<BOS><SYS>Personalidade e instruções da AYRIS.<USR>Oi, tudo bem?<AST>"
```

A aceitação de marcadores estruturais no argumento `--prompt` depende de como `generate.py` tokeniza o texto. Verifique o comportamento do script local antes de interpretar uma geração malformada. Para uma avaliação conversacional consistente, compare os checkpoints com os mesmos prompts, parâmetros de geração e seed.

Conjunto sugerido de avaliação manual:

- Cumprimentos e conversa casual.
- Continuidade com duas ou mais mensagens anteriores.
- Perguntas técnicas simples.
- Pedidos que exigem esclarecer informações ausentes.
- Situações em que a AYRIS deve admitir que não sabe.
- Mudança de assunto durante uma conversa.
- Conversas com assuntos e formulações que não aparecem no conjunto de treino.

Não confundir fluência, personalidade ou capacidade de manter turnos com conhecimento factual ou raciocínio confiável.

---

## 9. Métricas e conceitos

### Cross Entropy

No pré-treinamento literário, a Cross Entropy é calculada sobre todos os tokens-alvo. No fine-tuning de chat, a mesma métrica é calculada apenas nas posições selecionadas pela máscara.

\[
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(y_i\mid x_{<i})
\]

`N` é a quantidade de posições efetivamente incluídas na loss. No treinador conversacional, `N` corresponde aos tokens de resposta selecionados pela máscara, incluindo o EOS final conforme a política atual.

### Perplexity

\[
\mathrm{PPL}=e^{\mathcal{L}}
\]

Menor perplexity indica maior probabilidade média atribuída aos alvos avaliados. Só compare valores diretamente quando tokenizer, corpus, máscara e protocolo de avaliação forem compatíveis. A PPL do corpus literário não deve ser comparada diretamente à PPL do dataset conversacional.


### Gradiente e Gradient Clipping

A norma L2 mede a magnitude total do gradiente:

\[
\lVert g \rVert_2 = \sqrt{\sum_j g_j^2}
\]

O Gradient Clipping limita a norma do gradiente a um valor máximo \(c\):

\[
g' = g \cdot \min\left(1,\frac{c}{\lVert g \rVert_2}\right)
\]

Na AYRIS, o limite utilizado é:

\[
c = 1.0
\]

Quando a norma do gradiente ultrapassa esse limite, seus valores são ajustados proporcionalmente. Caso contrário, o gradiente permanece inalterado.

O objetivo é evitar atualizações excessivamente grandes durante o treinamento e melhorar a estabilidade da otimização.


### AdamW

AdamW mantém médias móveis dos gradientes e aplica weight decay desacoplado. Os valores usados nos treinadores atuais incluem betas `(0.9, 0.95)` e epsilon `1e-8`.

### Learning rate

Os treinadores utilizam warmup seguido de decaimento cosseno. Cada execução registra seu próprio scheduler. `--init-from` inicia um scheduler novo; `--resume` restaura o scheduler da execução existente.

---

## 10. Estrutura relevante do projeto

A estrutura abaixo reúne os diretórios e arquivos importantes conhecidos. O filesystem local pode conter outros scripts e diretórios experimentais.

```text
AYRIS/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   ├── tokenized/
│   └── chat/
│       ├── conversations.txt
│       ├── raw/
│       ├── splits/
│       └── tokenized/
├── docs/
├── experiments/
├── implementations/
├── monitor/
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── tokenize_corpus.py
│   │   ├── prepare_chat_dataset.py
│   │   └── chat_dataset.py
│   ├── inference/
│   │   └── generate.py
│   ├── model/
│   │   ├── embedding.py
│   │   ├── rope.py
│   │   ├── attention.py
│   │   ├── transformer_block.py
│   │   └── ayris.py
│   ├── tokenizer/
│   │   └── train_bpe.py
│   └── training/
│       ├── step.py
│       ├── trainer.py
│       ├── chat_step.py
│       └── chat_trainer.py
├── tests/
├── tools/
├── checkpoints/
│   ├── tokenizer/
│   └── training/
├── requirements.txt
└── README.md
```

Confirme a existência dos arquivos localmente antes de usar esta árvore como inventário exato. Em particular, confirme que `src/training/chat_trainer.py` foi salvo com o código experimental correspondente antes de executar os comandos de fine-tuning.

---

## 11. Boas práticas para continuar o projeto

1. Não sobrescrever checkpoints literários ao executar fine-tuning conversacional.
2. Dar a cada experimento um `run-name` exclusivo.
3. Usar `--init-from` para iniciar uma nova etapa e `--resume` somente para continuar a mesma execução.
4. Não alterar tokenizer depois de tokenizar os datasets sem reconstruir os arquivos tokenizados e verificar os hashes.
5. Manter treino, validação e teste separados por conversa completa no dataset de chat.
6. Não usar o split `test` para escolher checkpoints ou ajustar hiperparâmetros.
7. Aumentar o dataset conversacional com variedade e revisão, em vez de simplesmente multiplicar exemplos quase idênticos.
8. Preservar uma cópia dos checkpoints anteriores antes de iniciar experimentos novos.
9. Comparar gerações com prompts fixos e os mesmos parâmetros de inferência.
10. Não interpretar uma redução da loss em um conjunto minúsculo como prova de capacidade conversacional generalizável.
11. Verificar os testes e o código real do repositório antes de substituir arquivos centrais.
12. Não iniciar duas execuções simultâneas usando o mesmo `run-name` ou diretório de checkpoint.

Comandos úteis:

```bash
# Ver os argumentos do treinador literário
python -m src.training.trainer --help

# Ver os argumentos do treinador conversacional
python -m src.training.chat_trainer --help

# Executar os testes unitários
python -m unittest discover -s tests -p "test_*.py" -v

# Inspecionar os checkpoints e seus tamanhos
find checkpoints/training -maxdepth 4 -type f -printf '%p %s bytes\n'
```

---

## 12. Próximas etapas

1. Confirmar que `src/training/chat_trainer.py` está salvo e que `--help` funciona.
2. Executar o piloto conversacional de 40 atualizações.
3. Inspecionar `metrics.jsonl`, `ayris_best.pt` e `ayris_latest.pt`.
4. Comparar o checkpoint literário e o checkpoint de chat usando os mesmos prompts.
5. Avaliar o split `test` separadamente, sem usá-lo para selecionar o checkpoint.
6. Revisar e ampliar o dataset conversacional antes de qualquer treinamento mais longo.
7. Considerar retenção de parte do corpus literário em experimentos futuros, caso o fine-tuning cause esquecimento de padrões gerais de escrita.
8. Definir formalmente a interface e a arquitetura do módulo `think` antes de implementar NAT.
9. Projetar o orquestrador e as Tool Calls como componentes de software separados do modelo de conversação.
10. Só depois avaliar mudanças de escala, arquitetura ou formato de distribuição do modelo.

---

## 13. Instruções para retomar o projeto em outra conversa

Ao continuar o desenvolvimento:

1. Ler este README e confirmar se ele ainda corresponde ao repositório local.
2. Inspecionar os arquivos reais antes de presumir que o código coincide com a documentação.
3. Confirmar os checkpoints:

   ```bash
   find checkpoints/training -maxdepth 4 -type f -printf '%p %s bytes\n'
   ```

4. Confirmar os testes:

   ```bash
   python -m unittest discover -s tests -p "test_*.py" -v
   ```

5. Confirmar o pipeline de chat:

   ```bash
   python -m src.data.prepare_chat_dataset
   ```

6. Não refazer tokenizer, embedding, RoPE, atenção ou Transformer sem evidência de erro.
7. Preservar os checkpoints literários e executar o fine-tuning em um `run-name` separado.
8. Fazer alterações pequenas: fornecer o arquivo completo, executar um teste curto e só então iniciar treinamento mais longo.
9. Distinguir resultados efetivamente observados de hipóteses e etapas planejadas.

O usuário prefere instruções diretas em português brasileiro, comandos completos e validação por etapas. Normalmente executa os comandos localmente e retorna as saídas para análise.
