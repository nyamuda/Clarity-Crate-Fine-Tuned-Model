# Science Text Summarization API

This API provides access to a fine-tuned version of **t5-small** for **scientific and medical text summarization**. Tailored for students, researchers, and professionals, this model is fine-tuned on the **PubMed dataset** to effectively condense research articles, and papers into clear, concise summaries.

## Model Description

This API uses a fine-tuned version of **t5-small**, optimized for summarizing scientific content.
You can view the model on Hugging Face here: [Model on Hugging Face](https://huggingface.co/nyamuda/extractive-summarization).

## Summarization Type

The model uses **extractive summarization**, which means it picks out key sentences from the original text to create clear, concise summaries that keep the main information. Instead of creating new phrases, it focuses on keeping the important ideas and language of the scientific content intact, providing accurate and reliable summaries.

## Intended Use Cases

The model is ideal for:

- Summarizing research papers, scientific articles, and educational materials
- Supporting quick comprehension of long, detailed scientific documents
- Serving as a valuable resource in both educational and research applications

## Training Details

The model was fine-tuned on the PubMed dataset, which provides high-quality biomedical abstracts and summaries. This training ensures that the API prioritizes both factual accuracy and relevance in its summaries.

## Deployment

This API can be integrated into various applications, including:

- Document summarization services
- Text analysis tools for scientific and medical content
- Educational platforms aimed at simplifying complex scientific materials

## API Documentation

For more information on how to use this API, please refer to the [API Documentation](https://clarity-crate-fine-tuned-model.onrender.com/docs).
