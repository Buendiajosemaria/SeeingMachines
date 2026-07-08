# Seeing Machines: Screenshot Companion

A multimodal retrieval system built over my personal Steam screenshot archive: 470 images
across 72 games and roughly fourteen years of playing. It answers questions like
"pull up all screenshots from Cyberpunk 2077", "show me a nighttime city street",
or "what game is this from?".

Final project for **CompSci for Designers 2**, MA Design for Digital Futures,
TH Nürnberg, Summer 2026. Submitted at **Level 3 (The Critic)**.

**Full documentation:** https://github.com/Buendiajosemaria/SeeingMachines/tree/main

**Netlify Mirror** https://seeingmachines.netlify.app/

**Google Drive to the Notebook** https://drive.google.com/file/d/1Os8bx9-f1QO8Q7Hqx4jLLS2RpIgDR15q/view?usp=drive_link

## What it does

Three retrieval routes run over the same corpus:

- **Level 1** embeds every image with CLIP (ViT-B/32) and searches by visual similarity.
  Game-name queries bypass the embeddings and filter on Steam App ID metadata, since the
  folder structure already guarantees that answer. Uploading an unsorted screenshot runs a
  nearest-neighbor identification against the archive.
- **Level 2** has Gemma 3 4B write a structured JSON description of every screenshot. The
  descriptions are embedded with a sentence-transformer (all-MiniLM-L6-v2) and searched as
  text. Retrieved captions feed back into Gemma for a grounded natural-language answer.
- **Level 3** fuses both similarity scores with a user-controlled weight (a slider in the
  interface), passes retrieved images directly into the VLM's context at answer time, and
  measures all three routes against two evaluation benchmarks.

The interface is a Gradio app with all five modes selectable, launched from the notebook.

## Running it

Open the notebook in Google Colab on a **T4 GPU runtime** and run all cells top to bottom.

Requirements:

- A HuggingFace account with the (free) Gemma license accepted at
  https://huggingface.co/google/gemma-3-4b-it
- A HuggingFace access token (read scope) for the login cell

Everything expensive is cached in this repository: CLIP embeddings, the caption JSON,
caption embeddings, and the App ID name mapping. A fresh runtime loads the caches and
recomputes nothing. The captioning cell checks the cache and skips itself, so a full run
costs a few minutes of setup instead of the roughly three GPU hours the caches represent.
The GPU is only needed to host Gemma for live answers. On a CPU runtime both retrieval
routes still work from cache and only the generated answers are unavailable.

## Repository layout

```
SeeingMachines/
  ORTEGA_seeing_machines_companion.ipynb   the notebook
  SeeingMachines/
    cache/
      appid_to_name.json                   App ID to game name mapping
      clip_embeddings.npy                  Level 1 image embeddings
      clip_embeddings_index.csv            row index for the embeddings
      captions.json                        Level 2 VLM descriptions (470 images)
      caption_embeddings_st.npy            Level 2 caption embeddings
      misseeing_truncation_failures.json   archived caption failures, kept as evidence
      evaluation_results.csv               hand-labeled benchmark results
      evaluation_gameid.csv                game-identity benchmark results
    screenshots/
      <appid>/screenshots/*.jpg            the image corpus, organized by Steam App ID
  index.html                               project documentation page
  images/                                  figures for the documentation page
```

## On the screenshots

The corpus consists of screenshots I took of commercial games. They are included here for
academic evaluation of this course project. The repository will be taken down after
grading. If you are a rights holder and want something removed sooner, open an issue.

## Findings, briefly

The documentation page has the full story, including a six-case dossier of documented
failures. The short version: 69% of the first captioning run truncated because my token
budget was too tight, six screenshots defeated the fixed budget by containing too much
on-screen text to transcribe, one caption failed on curly quotes alone, and the caption
retrieval route initially collapsed into a handful of hub images because I reused CLIP's
text encoder for a job it was never trained to do. Every failure has a mechanism, a fix,
and a lesson. The evaluation shows the two base routes succeed on different species of
query, which is the argument for the fusion slider existing at all.

## Credits

Built with OpenCLIP, Gemma 3 4B (4-bit via bitsandbytes), sentence-transformers, Gradio,
and the free tier of Google Colab. Developed in collaboration with AI assistants, mainly
Claude; all judgment calls, the description schema, the corpus, and the evaluation labels
are mine. Course taught by Moritz Schwind and Christopher Kopic.
