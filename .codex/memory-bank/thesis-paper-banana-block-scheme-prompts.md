# Paper-Banana Prompts For Thesis Block Schemes

Use these prompts to regenerate the thesis block-scheme figures. The text inside every figure
must be in Russian. Keep the target base filenames unchanged; the LaTeX files use extensionless
`\includegraphics`, so PNG/JPG/WebP files with these names can replace the current PDFs.
When the generated raster files are ready, remove or move aside the old PDF block-scheme files with
the same base names; otherwise LaTeX may keep choosing the PDF version first.

## `experiment_pipeline`

Create a clean academic block diagram for a Russian master's thesis about EEG-based reconstruction
of random 6x6 binary visual imagery. White background, flat vector style, thin dark-gray outlines,
subtle muted colors, no gradients, no shadows, no 3D, no decorative elements. Text language:
Russian only. Layout: horizontal main pipeline with three small control blocks below it.

Main row blocks from left to right:
1. "EEG-эпоха 15 с"
2. "Представления: признаки / спектр"
3. "Модели: классика / CNN"
4. "36 логитов"
5. "Матрица 6 x 6"

Lower control blocks:
under representation block: "Нормировка только по обучению";
under model block: "Групповая валидация";
under output blocks: "Кластерный bootstrap".

Use simple arrows between the main blocks and thin arrows from relevant main blocks to lower
control blocks. The diagram should fit an A4 thesis page width, be readable when printed in
grayscale, and leave generous margins. Negative prompt: no English labels, no code paths, no
icons, no neural-network artwork, no EEG scalp image, no tiny text, no dark background, no
watermark.

## `eegnet_architecture`

Create a clean academic architecture block diagram for a Russian master's thesis. The figure shows
the spectral-input adaptation of EEGNet for multi-label reconstruction of a 6x6 image. White
background, flat vector style, restrained muted colors, thin dark-gray outlines, consistent
sans-serif typography. Text language: Russian only. Layout: one horizontal chain of blocks with
arrows. No equations except the input shape.

Blocks from left to right:
1. "Вход B x P x C x W"
2. "Темпоральная свёртка"
3. "Depthwise-свёртка по электродам"
4. "ELU, average pooling, dropout"
5. "Separable-свёртка"
6. "Глобальный pooling"
7. "36 логитов"

Add a small subtitle below the chain: "Спектральная адаптация EEGNet; sigmoid и порог применяются
после модели". Keep all labels large and readable. Negative prompt: no English-only figure, no
Keras/TensorFlow logo, no PyTorch logo, no layer sizes invented beyond the listed labels, no
photorealism, no decorative gradients, no tiny unreadable text, no watermark.

## `deepconvnet_architecture`

Create a clean academic architecture block diagram for a Russian master's thesis. The figure shows
the spectral-input adaptation of DeepConvNet for multi-label reconstruction of a 6x6 image. White
background, flat vector style, restrained muted colors, thin dark-gray outlines, consistent
sans-serif typography. Text language: Russian only. Layout: one horizontal chain of blocks with
arrows. No equations except the input shape.

Blocks from left to right:
1. "Вход B x P x C x W"
2. "Темпоральная свёртка 25"
3. "Пространственная свёртка 25"
4. "Conv-блок 50"
5. "Conv-блок 100"
6. "Conv-блок 200"
7. "Глобальный pooling"
8. "36 логитов"

For the three Conv-блок blocks, add a tiny repeated note inside or below them: "BN + ELU + max
pool + dropout". Keep the figure compact enough for A4 width and readable in grayscale. Negative
prompt: no English-only labels, no unlisted layers, no invented accuracy values, no code paths, no
photorealism, no 3D, no decorative gradients, no watermark.

## `shallowconvnet_architecture`

Create a clean academic architecture block diagram for a Russian master's thesis. The figure shows
the spectral-input adaptation of ShallowConvNet for multi-label reconstruction of a 6x6 image.
White background, flat vector style, restrained muted colors, thin dark-gray outlines, consistent
sans-serif typography. Text language: Russian only. Layout: one horizontal chain of blocks with
arrows. No equations except the input shape.

Blocks from left to right:
1. "Вход B x P x C x W"
2. "Темпоральная свёртка 40"
3. "Пространственная свёртка 40"
4. "Квадратичная нелинейность"
5. "Average pooling"
6. "Логарифмирование"
7. "Dropout"
8. "Глобальный pooling"
9. "36 логитов"

Add a small subtitle below the chain: "Схема отражает band-power-подобную логику: фильтрация,
оценка мощности, логарифмирование". Keep all text readable after insertion into a thesis page.
Negative prompt: no English-only labels, no unlisted layers, no code paths, no invented metrics,
no photorealism, no 3D, no decorative gradients, no watermark.
