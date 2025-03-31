from typing import List, Optional, Union
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import requests

class TextEmbedder:
    """A class to handle text embedding operations using OpenAI's API"""

    def __init__(self):
        """
        Initialize the TextEmbedder.

        Args:
            model (str): The OpenAI embedding model to use.
                        Defaults to "text-embedding-3-large".
        """
        load_dotenv()
 

    def get_reference_dictionary(self, texts: List[str]):
        reference_dict = {}
        for text in texts:
            reference_dict[text] = self.get_embedding(text)
        return reference_dict        



    async def get_embedding(self,
                    text: Union[str, List[str]],
                    normalize: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get embeddings for one or more texts using remote HTTP API.

        Args:
            text (Union[str, List[str]]): Single text string or list of text strings
                                        to get embeddings for.
            normalize (bool): Whether to normalize the resulting vectors.
                            Defaults to False.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: The embedding vector(s) as numpy array(s).
            For single text input, returns a single numpy array.
            For list input, returns a list of numpy arrays.

        Raises:
            Exception: If there's an error calling the API.
        """
        # Convert single string to list for consistent processing
        input_texts = [text] if isinstance(text, str) else text

        try:
            print(f"Calling embedding API for {len(input_texts)} text embedding(s).")
            
            embeddings = []
            for text in input_texts:
                payload = {
                    "text": text,
                    "normalize": normalize
                }
                
                response = await requests.post(
                    os.getenv("REMOTE_API_URL"),
                    json=payload
                )
                
                if response.status_code != 200:
                    raise Exception(f"API request failed with status code: {response.status_code}")
                    
                response_data = response.json()
                embedding = np.array(response_data["embedding"], dtype=np.float32)
                embeddings.append(embedding)

            # Return single array for single input, list for multiple inputs
            return embeddings[0] if isinstance(text, str) else embeddings

        except Exception as e:
            raise Exception(f"Error creating embedding: {e}")

