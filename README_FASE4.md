# AYRIS — Fase 4: treinamento por épocas (batch 3)

Este pacote substitui `src/training/step.py` e `src/training/trainer.py`.
Ele mantém a arquitetura existente e altera o pipeline de treinamento para:

- batch size padrão 3;
- 100.000 atualizações por padrão;
- embaralhamento sem reposição dentro de cada época;
- persistência da ordem da época e da posição para retomada;
- alerta vermelho de possível overfitting;
- registro de duração, velocidade, tokens/s e ETA;
- checkpoints atômicos contendo modelo, AdamW, scheduler, RNG e sampler;
- validação fixa com 128 janelas a cada 5.000 passos;
- checkpoint recente a cada 1.000 passos.

## Instalação

Na raiz do repositório AYRIS, extraia o ZIP preservando os caminhos. Os arquivos em `src/training/` serão substituídos e o teste `tests/test_epoch_sampler.py` será adicionado.

## Validar antes de treinar

```bash
python -m py_compile src/training/step.py src/training/trainer.py
python -m unittest discover -s tests -p "test_*.py" -v
```

Os testes novos verificam que cada índice aparece uma única vez por época, que as épocas são reembaralhadas e que a retomada preserva exatamente o próximo lote.

## Iniciar fase 4 a partir do melhor checkpoint da fase 3

```bash
python -m src.training.trainer \
  --steps 100000 \
  --batch-size 3 \
  --init-from checkpoints/training/fase_03_5000/ayris_best.pt \
  --run-name fase_04_100k \
  --threads 4
```

O `--init-from` carrega apenas os pesos do checkpoint anterior. AdamW e scheduler começam novos para a fase 4.

## Retomar após Ctrl+C ou interrupção

Use os mesmos hiperparâmetros e o mesmo nome da execução:

```bash
python -m src.training.trainer \
  --steps 100000 \
  --batch-size 3 \
  --resume \
  --run-name fase_04_100k \
  --threads 4
```

A retomada rejeita alterações de configuração e alterações nos dados/tokenizer, evitando continuar com estado incompatível.

## Saídas

```text
checkpoints/training/fase_04_100k/
├── ayris_best.pt
├── ayris_latest.pt
└── metrics.jsonl
```

- `ayris_best.pt`: menor Validation Loss observada nesta fase, incluindo o checkpoint inicial da fase.
- `ayris_latest.pt`: estado mais recente para retomada.
- `metrics.jsonl`: eventos de treinamento, validação, checkpoints e tempos.

## Épocas e quantidade de dados

O conjunto atual contém 205.482 janelas. Com batch 3, uma época contém:

`ceil(205482 / 3) = 68494` atualizações.

Como 205.482 é divisível por 3, todas as atualizações desta configuração usam exatamente 3 janelas. Cada época utiliza todas as janelas uma vez, em ordem aleatória. A ordem é persistida no checkpoint.

100.000 passos × 3 sequências × 512 tokens = 153.600.000 posições de alvo processadas. Há repetição entre épocas, mas não dentro da mesma época.

## Alerta de possível overfitting

Em cada avaliação, o treinador compara a Validation Loss com o melhor valor visto nesta fase. Se ficar mais de 1% acima do melhor em duas avaliações consecutivas, imprime uma mensagem ANSI vermelha:

`[ALERTA] POSSÍVEL OVERFITTING`

É um alerta heurístico para investigar a tendência, não uma prova definitiva de overfitting. Uma nova melhora na validação zera a contagem.

## Registro de tempo

O terminal e `metrics.jsonl` registram:

- duração acumulada da fase;
- velocidade média em passos/s e tokens/s;
- ETA aproximado;
- timestamp UTC dos eventos;
- tempo acumulado salvo no checkpoint e recuperado no `--resume`.

O tempo total inclui sessões anteriores quando uma execução é retomada.

## Configuração padrão

| Parâmetro | Valor |
|---|---:|
| Passos | 100.000 |
| Batch | 3 |
| Contexto | 512 |
| LR máximo | 0,0001 |
| Warmup | 1.000 passos |
| LR mínimo | 0,00001 |
| Scheduler | Cosseno |
| AdamW betas | (0,9; 0,95) |
| Weight decay | 0,1 em matrizes; 0 em vetores |
| Gradient clipping | 1,0 |
| Validação | 128 janelas a cada 5.000 passos |
| Checkpoint recente | A cada 1.000 passos |
| Log | A cada 250 passos |
| Precisão | FP32 |
| Dispositivo | CPU |

## Observações operacionais

- A validação periódica usa 128 janelas fixas, para comparação consistente entre checkpoints. Faça uma avaliação completa de validação e teste separadamente ao fim da fase.
- O conjunto de teste não é usado pelo treinador.
- Monitore temperatura, RAM e espaço em disco durante a execução.
- O batch 3 aumenta a memória necessária em relação ao batch 1; se houver pressão de memória, interrompa e planeje uma configuração menor em uma nova execução.
- Os checkpoints incluem estados do AdamW e, por isso, são consideravelmente maiores que arquivos contendo somente os pesos.
