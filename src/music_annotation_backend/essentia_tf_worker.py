"""Isolated macOS adapter; identical standard C++ frontends and tensor layouts."""
import math
import numpy as np

def predict(pool, graph, inputs, outputs, squeeze=True):
    import essentia.standard as es
    return es.TensorflowPredict(graphFilename=graph, inputs=[inputs], outputs=outputs, squeeze=squeeze)(pool)

def backbone(audio, family, graph, hop_frames):
    import essentia
    import essentia.standard as es
    if family not in {"musicnn", "effnet", "tempo"} or not 1 <= hop_frames <= 2048: raise ValueError("Invalid TF options")
    tempo, effnet = family == "tempo", family == "effnet"
    frame, hop = (1024, 512) if tempo else (512, 256)
    patch, width = (256, 40) if tempo else (128 if effnet else 187, 96)
    frontend = es.TensorflowInputTempoCNN() if tempo else es.TensorflowInputMusiCNN()
    mel = np.asarray([frontend(raw) for raw in es.FrameGenerator(audio, frameSize=frame, hopSize=hop, startFromZero=False, validFrameThresholdRatio=0)], dtype=np.float32)
    count = math.ceil(len(audio) / (hop * hop_frames))
    output_width = 256 if tempo else 1280 if effnet else 200
    if not len(mel) or count > 11000 or count * output_width > 2000000: raise ValueError("Invalid TF point count; increase prediction interval")
    inputs = "input" if tempo else "serving_default_melspectrogram" if effnet else "model/Placeholder"
    output = "output" if tempo else "PartitionedCall:1" if effnet else "model/dense/BiasAdd"
    outputs = [output] + (["model/Sigmoid"] if family == "musicnn" else [])
    model = es.TensorflowPredict(graphFilename=graph, inputs=[inputs], outputs=outputs, squeeze=not tempo)
    rows, tags = [], []
    for offset in range(0, count, 64):
        actual = min(64, count - offset)
        batch = 64 if effnet else actual
        tensor = np.zeros((batch, 1, patch, width), dtype=np.float32)
        for b in range(actual):
            start = max(0, min(math.floor((offset + b + .5) * hop_frames - patch / 2 + .5), max(0, len(mel) - patch)))
            tensor[b, 0] = mel[start + np.arange(patch) % (len(mel) - start)]
        if tempo:
            mean = tensor.mean(axis=(1, 2, 3), keepdims=True, dtype=np.float64)
            std = tensor.std(axis=(1, 2, 3), keepdims=True, dtype=np.float64)
            tensor = np.asarray((tensor - mean) / np.where(std == 0, 1, std), dtype=np.float32).transpose(0, 3, 2, 1).copy()
        pool = essentia.Pool(); pool.set(inputs, tensor)
        result = model(pool)
        rows.extend(np.asarray(result[output]).reshape(batch, -1)[:actual].tolist())
        if family == "musicnn": tags.extend(np.asarray(result["model/Sigmoid"]).reshape(batch, -1)[:actual].tolist())
    return {"protocol": 1, "matrix": rows, **({"tags": tags} if family == "musicnn" else {})}

def head(values, graph, inputs, output, width, output_width):
    import essentia
    import essentia.standard as es
    if not 1 <= width <= 1280 or not 1 <= output_width <= 400 or len(values) % width: raise ValueError("Invalid TF head size")
    rows = values.reshape(-1, width)
    if len(rows) > 11000: raise ValueError("Too many TF rows")
    model = es.TensorflowPredict(graphFilename=graph, inputs=[inputs], outputs=[output])
    result = []
    for offset in range(0, len(rows), 64):
        batch = rows[offset:offset + 64]
        pool = essentia.Pool(); pool.set(inputs, np.asarray(batch[:, None, None, :], dtype=np.float32))
        result.extend(np.asarray(model(pool)[output]).reshape(len(batch), output_width).tolist())
    return {"protocol": 1, "matrix": result}
