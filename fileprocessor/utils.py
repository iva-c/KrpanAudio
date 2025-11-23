import os
import re
import base64
import json

from typing import Tuple, Dict, Any
from openai import OpenAI

from .classic_parsing import classic_doc_parsing

def process_file(
    input_file: Any,
    mode: str,
    output: str,
    api_key: str,
    output_file_path: str,
    pdf_method: str = None
) -> str:
    """
    Processes an input document file, extracts text and images, describes images, integrates descriptions,
    and outputs the result as text or audio.

    Args:
        input_file (Any): The input file object (PDF or DOCX).
        mode (str): Integration mode ('separate' for direct replacement, 'integrated' for LLM-based merging).
        output (str): Output format ('text' or 'audio').
        api_key (str): API key for accessing the LLM service.
        output_file_path (str): Path to save the output file.
        pdf_method (str, optional): Document parsing method ('classic' or 'llm').

    Returns:
        str: Status message describing the processing result and output location.
    """

    output_dir = os.path.dirname(output_file_path)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # parse input file into text and images
        if not api_key or pdf_method == "classic":
            text, images = classic_doc_parsing(input_file.temporary_file_path())
            if not api_key:
                for img_name, img_data in images.items():
                    img_path = os.path.join(os.path.dirname(output_file_path), img_name)
                    with open(img_path, 'wb') as img_file:
                        img_file.write(base64.b64decode(img_data))
                return f"Processed {input_file.name} to {output_file_path} with images saved as PNGs."
        elif pdf_method == "llm":
            try:
                text, images = llm_doc_parsing(input_file.temporary_file_path(), api_key)
            except Exception as e:
                return f"Error during LLM document parsing: {str(e)}"
    except Exception as e:
        return f"Error during document parsing: {str(e)}"

    try:
        descriptions = describe_images_with_llm(images, text, api_key)
    except Exception as e:
        return f"Error during image description: {str(e)}"

    try:
        if mode == 'separate':
            integrated_text = text
            for img_tag, description in descriptions.items():
                integrated_text = integrated_text.replace(f'<{img_tag}>', description)
        elif mode == 'integrated':
            integrated_text = integrate_descriptions_with_llm(text, descriptions, api_key)
    except Exception as e:
        return f"Error during description integration: {str(e)}"

    try:
        if output == "text":
            with open(output_file_path, 'w', encoding='utf-8') as out_file:
                out_file.write(integrated_text)
        elif output == "audio":
            audio_data = text_to_audio(integrated_text, api_key)
            with open(output_file_path, "wb") as audio_file:
                audio_file.write(audio_data)
    except Exception as e:
        return f"Error during output file writing: {str(e)}"

    return f"Processed {input_file.name} with mode {mode}, outputting to {output_file_path}"

def llm_doc_parsing(doc_path: str, api_key: str) -> Tuple[str, Dict[str, str]]:
    """
    Processes the given document using LLM-based parsing.

    Args:
        doc_path (str): Path to the input document (PDF or DOCX).

    Returns:
        tuple:
            text (str): The extracted text content from the document.
            images (dict): A dictionary where keys are image filenames and values are base64-encoded PNG image data.
    """

    # TODO, če je file docx ali pdf s text in slikami, uporabi classic parsing
    try:
        client = OpenAI(api_key=api_key)
        # Make a minimal test call to check the key (e.g., list models)
        client.models.list()
    except Exception as e:
        raise RuntimeError(f"Invalid OpenAI API key or connection error: {str(e)}")
    
    with open(doc_path, "rb") as f:
        file_data = f.read()
    base64_string = base64.b64encode(file_data).decode("utf-8")

    system_prompt = (
        "Si asistent za predelovanje pdf datotek v txt besedilo in slike. Tvoja naloga je, "
        "da v vhodni pdf datoteki prepoznaš besedilo in slike in jih pretvoriš v txt datoteko "
        "in png datoteke. Da bi ohranili sledljivost pozicij slik, povsod, kjer najdeš sliko, "
        "v txt datoteki označi z <slika_x> kjer je x številka slike. Sliko iz tega mesta označi "
        "z slika_x in jo shrani v png datoteko. "
    )
    try:
        response = client.responses.create(
            model="gpt-5",
            input=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": doc_path.split("/")[-1],
                            "file_data": f"data:application/pdf;base64,{base64_string}",
                        }
                    ],
                },
            ]
        )
    except Exception as e:
        raise RuntimeError(f"LLM document parsing failed: {str(e)}")

    output = response.output_text
    try:
        result = json.loads(output)
        text = result.get("text", "")
        images = result.get("images", {})
    except Exception:
        text = output
        images = {}

    return text, images

def describe_images_with_llm(
    images: Dict[str, str],
    text: str,
    api_key: str,
) -> Dict[str, str]:
    """
    Describes images using LLMs based on the provided text context.

    Args:
        images (dict): A dictionary where keys are image filenames and values are base64-encoded image data.
        text (str): The whole input file text.
        api_key (str): API key for accessing the LLM service.

    Returns:
        dict: A dictionary where keys are image filenames and values are their descriptions.
    """

    client = OpenAI(api_key=api_key)
    descriptions = {}
    image_tags = re.findall(r'<(image\d+)>', text)

    for tag in image_tags:
        tag_pos = text.find(f'<{tag}>')
        if tag_pos == -1:
            continue

        words = text.split()
        tag_word_index = len(text[:tag_pos].split())
        start = max(0, tag_word_index - 500)
        end = min(len(words), tag_word_index + 500)
        context = ' '.join(words[start:end])

        image_data = images.get(tag, None)
        if not image_data:
            descriptions[tag] = "No image data found."
            continue

        system_prompt = ("Si strokovnjak za opisovanje slik v knjigah slepim osebam. V user promptu "
                        "boš dobil sliko in besedilo iz knjige, ki sliko obdaja. Iz slike izlušči podrobnosti in "
                        "informacije, ki jih slepa oseba potrebuje, da bi razumela, kaj prikazuje ter vsebino slike "
                        "opiši v slovenščini. Dolžina opisa naj bo en odstavek, besedilo naj bo v istem slogu kot "
                        "besedilo. Cilj tega opisa je obogatiti izkušnjo slepe osebe. Upoštevaj, da bi ta opis nadomestil "
                        "sliko, zato ne sklepaj, analiziraj ali domnevaj, kaj se dogaja na sliki, in ne omenjaj "
                        "subjektivnih čustev ali občutkov. Osredotoči se le na opis upodobljenega in sliko čimbolje "
                        "opiši za ta namen.")
        
        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": context},
                            {"type": "input_image", "image_url": f"data:image/png;base64,{image_data}"},
                        ],
                    }
                ],
            )
        except Exception as e:
            descriptions[tag] = f"Error during image description: {str(e)}"
            continue

        
        descriptions[tag] = response.output_text.strip()

    return descriptions

def integrate_descriptions_with_llm(
    text: str,
    descriptions: Dict[str, str],
    api_key: str,
) -> str:
    """
    Integrates image descriptions into the text using LLMs.

    Args:
        text (str): The whole input file text.
        descriptions (dict): A dictionary where keys are image filenames and values are their descriptions.
        api_key (str): API key for accessing the LLM service.

    Returns:
        str: The text with integrated image descriptions.
    """
    client = OpenAI(api_key=api_key)
    integrated_text = text
    image_tags = re.findall(r'<(image\d+)>', text)

    for tag in image_tags:
        tag_pos = integrated_text.find(f'<{tag}>')
        if tag_pos == -1:
            continue

        words = integrated_text.split()
        tag_word_index = len(integrated_text[:tag_pos].split())
        start = max(0, tag_word_index - 500)
        end = min(len(words), tag_word_index + 500)
        context = ' '.join(words[start:end])

        #TODO should add a check to see if images have at least 500 words between them
        # if they are not, they sould be processed together

        description = descriptions.get(tag, "")

        system_prompt = (
            "Si strokovnjak, ki prilagaja knjige s slikami v besedilo za slepe osebe. "
            "Tvoja naloga je, da opise slik integriraš v besedilo in ga dopolniš tako, "
            "da je ne moti toka besedila in ga neopazno dopolnjuje. V user promptu boš "
            "dobil opis slike ter besedilo, ki jo obdaja. Opis slike vključi v besedilo "
            "v user promptu. Bodi pozoren na to, da besedilo obogatiš samo z informacijami "
            "iz opisa slike.  "
        )

        user_content = [
            {"type": "input_text", "text": f"Context:\n{context}\n\nDescription:\n{description}"}
        ]
        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    }
                ],
            )
        except Exception as e:
            raise RuntimeError(f"Error during description integration for {tag}: {str(e)}")

        updated_context = response.output_text.strip()
        integrated_text = integrated_text.replace(f'<{tag}>', updated_context)

    return integrated_text


def text_to_audio(
    text: str,
    api_key: str,
    chunk_size: int = 3000,
) -> bytes:
    """
    Converts text to audio using LLM-based synthesis, chunking for large documents at sentence boundaries.

    Args:
        text (str): The text to convert to audio.
        api_key (str): API key for accessing the LLM service.
        chunk_size (int): Maximum number of characters per chunk (default: 3000).

    Returns:
        bytes: The merged audio data.
    """

    client = OpenAI(api_key=api_key)
    system_prompt = ( 
        "You are a professional audiobook narrator. "
        "Read the provided text in a soft, warm, and welcoming female voice suitable for audiobooks. "
        "Return the audio in a standard format."
    )

    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())

    audio_segments = []

    for chunk in chunks:

        try:
            response = client.audio.create(
                model="tts-1",
                input=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": chunk}
                        ],
                    }
                ],
                response_format="mp3"
            )
        except Exception as e:
            raise RuntimeError(f"Text-to-audio conversion failed: {str(e)}")
        
        audio_segments.append(response.audio_data)

    merged_audio = b"".join(audio_segments)
    return merged_audio