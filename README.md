# Detecção de Mãos com Mediapipe

Este projeto implementa um sistema de detecção de mãos utilizando a biblioteca Mediapipe e OpenCV. Ele captura vídeo da webcam, detecta os pontos de referência da mão e visualiza os movimentos desenhando vetores e calculando os ângulos entre eles.

## Estrutura do Projeto

```
src
├── __init__.py
├── main.py
├── hand_detection.py
├── hand_detection_gui.py
├── calculation_amplitude.py
└── vector_drawer.py
requirements.txt
README.md
```

## Instalação

Para configurar o projeto, certifique-se de ter o Python instalado em sua máquina. Em seguida, instale as dependências necessárias executando:

```
pip install -r requirements.txt
```

## Uso

Para executar a aplicação de detecção de mãos, rode o seguinte comando:

```
python src/main.py
```

Isso abrirá uma janela exibindo o vídeo da webcam com os pontos de referência e vetores desenhados sobre a mão. Pressione a tecla `Esc` para encerrar a aplicação.

## Dependências

O projeto requer os seguintes pacotes Python:

- `mediapipe`
- `opencv-python`

Você pode instalar esses pacotes usando o arquivo `requirements.txt` fornecido.
