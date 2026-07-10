import hashlib
import io
import shutil
import numpy as np
from pathlib import Path
from typing import List, Tuple

import fitz
import torch
from PIL import Image

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

from config import *
from prompts import *

class DocumentProcessor:
    """
    Multimodal RAG Processor

    - Text embeddings -> OpenAI
    - Image embeddings -> CLIP
    - Separate Chroma collections
    """

    def __init__(
        self,
        docs_path: str = "docs",
        chroma_text_path: str = "./chroma_text",
        chroma_image_path: str = "./chroma_images"
    ):

        self.docs_path = Path(docs_path)

        self.chroma_text_path = Path(chroma_text_path)
        self.chroma_image_path = Path(chroma_image_path)

        print("🔤 Loading text embedding model...")
        self.text_embeddings = OpenAIEmbeddings(
            model=EMBEDDINGS_MODEL
        )

        print("🖼️ Loading CLIP image embedding model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)

        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ".",
                "!",
                "?",
                ",",
                " ",
                ""
            ]
        )

    ###########################################################################
    # TEXT DOCUMENTS
    ###########################################################################

    def load_documents(self) -> List[Document]:

        print(f"📚 Loading PDFs from {self.docs_path}")

        loader = PyPDFDirectoryLoader(str(self.docs_path))

        documents = loader.load()

        for doc in documents:

            filename = Path(doc.metadata["source"]).stem

            doc.metadata.update({
                "filename": filename,
                "doc_type": self._get_doc_type(filename),
                "doc_id": self._generate_doc_id(doc.page_content),
                "modality": "text"
            })

        print(f"✅ Loaded {len(documents)} text pages")

        return documents

    def split_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:

        print("✂️ Splitting text into chunks...")

        chunks = self.text_splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):

            chunk.metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk.page_content)
            })

        print(f"✅ Created {len(chunks)} text chunks")

        return chunks

    ###########################################################################
    # IMAGE EXTRACTION
    ###########################################################################


    def extract_images_from_pdfs(self) -> List[Document]:

        print("🖼️ Extracting images from PDFs...")

        image_documents = []

        pdf_files = list(self.docs_path.glob("*.pdf"))

        # Global duplicate tracking

        MIN_WIDTH = 100
        MIN_HEIGHT = 100

        ###############################
        debug_dir = Path("debug_extracted_images")

        accepted_dir = debug_dir / "accepted"
        #rejected_dir = debug_dir / "rejected"
        accepted_dir.mkdir(parents=True, exist_ok=True)
        #rejected_dir.mkdir(parents=True, exist_ok=True)
        ###############################
        for pdf_path in pdf_files:

            processed_xrefs = set()
            seen_hashes = set()
            accepted_embeddings = []

            print(f"📄 Processing images from: {pdf_path.name}")

            try:
                pdf = fitz.open(str(pdf_path))

                for page_index in range(len(pdf)):

                    page = pdf[page_index]

                    images = page.get_images(full=True)

                    for image_index, img in enumerate(images):

                        try:

                            xref = img[0]

                            ###################################################
                            # Skip repeated PDF image references
                            ###################################################

                            if xref in processed_xrefs:
                                continue

                            processed_xrefs.add(xref)

                            ###################################################
                            # Extract image
                            ###################################################

                            base_image = pdf.extract_image(xref)

                            image_bytes = base_image["image"]

                            ###################################################
                            # Exact duplicate detection
                            ###################################################

                            image_hash = hashlib.md5(
                                image_bytes
                            ).hexdigest()

                            if image_hash in seen_hashes:
                                continue

                            seen_hashes.add(image_hash)

                            ###################################################
                            # Load image
                            ###################################################

                            pil_image = Image.open(
                                io.BytesIO(image_bytes)
                            ).convert("RGB")

                            ###################################################
                            # Remove tiny images
                            ###################################################

                            width, height = pil_image.size

                            if (
                                width < MIN_WIDTH
                                or height < MIN_HEIGHT
                            ):
                                continue

                            ###################################################
                            # Save accepted image
                            ###################################################

                            filename = (
                                f"{pdf_path.stem}"
                                f"_page{page_index+1}"
                                f"_xref{xref}.png"
                            )

                            save_path = accepted_dir / filename

                            #pil_image.save(save_path)
                            if not save_path.exists():
                                continue
                            
                            pil_image.thumbnail((128, 128))
                            ###################################################
                            # Compute embedding
                            ###################################################

                            embedding = self.embed_image(
                                pil_image
                            )

                            ###################################################
                            # Generate pathology caption
                            ###################################################

                            caption = self.describe_image(
                                pil_image
                            )
                            ###################################################
                            # Create document
                            ###################################################

                            image_doc = Document(
                                page_content=caption,
                                metadata={
                                    "source": str(pdf_path),
                                    "filename": pdf_path.name,
                                    "page": page_index + 1,
                                    "image_index": image_index,
                                    "xref": xref,
                                    "width": width,
                                    "height": height,
                                    "modality": "image",
                                    "caption": caption,
                                    "embedding": embedding
                                }
                            )

                            image_documents.append(
                                image_doc
                            )

                            print(
                                f"✅ Image added | "
                                f"{pdf_path.name} | "
                                f"page={page_index+1} | "
                                f"size={width}x{height}"
                            )
                            print(caption)

                        except Exception as e:

                            print(
                                f"⚠️ Error extracting image "
                                f"{image_index}: {e}"
                            )

            except Exception as e:

                print(
                    f"⚠️ Error processing PDF "
                    f"{pdf_path.name}: {e}"
                )

        print(
            f"\n✅ Extracted "
            f"{len(image_documents)} unique images"
        )

        return image_documents

    ###########################################################################
    # IMAGE EMBEDDINGS
    ###########################################################################
    def embed_image(self, image: Image.Image):

        inputs = self.clip_processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():

            outputs = self.clip_model.get_image_features(
                **inputs
            )

        # Some transformers versions return tensor directly
        if isinstance(outputs, torch.Tensor):

            image_features = outputs

        else:
            # Older/newer compatibility
            image_features = outputs.pooler_output

        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        )

        return image_features[0].cpu().numpy().tolist()
    
    def embed_query_image_text(self, text: str):

        inputs = self.clip_processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():

            outputs = self.clip_model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

            text_features = outputs.pooler_output

        text_features = F.normalize(
            text_features,
            p=2,
            dim=-1
        )

        return text_features[0].cpu().numpy().tolist()

    def describe_image(
        self,
        image: Image.Image
    ) -> str:

        import base64
        from io import BytesIO
        from openai import OpenAI

        client = OpenAI()

        buffer = BytesIO()
        image.save(buffer, format="PNG")

        image_b64 = base64.b64encode(
            buffer.getvalue()
        ).decode()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
    Describe only visible elements in this image.

    Focus on:
    - plant structures
    - lesions
    - chlorosis
    - necrosis
    - spots
    - texture
    - color patterns
    - canopy condition
    - disease-related visual symptoms

    Do not diagnose.

    Return a concise scientific description
    (2-4 sentences).
    """
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ]
        )

        return response.choices[0].message.content.strip()
    ###########################################################################
    # VECTORSTORES
    ###########################################################################

    def create_text_vectorstore(
        self,
        documents: List[Document]
    ) -> Chroma:

        print("🔄 Creating TEXT ChromaDB...")

        if self.chroma_text_path.exists():
            shutil.rmtree(self.chroma_text_path)

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.text_embeddings,
            persist_directory=str(self.chroma_text_path),
            collection_name="disease_text_knowledge"
        )

        print(
            f"✅ Text vectorstore created "
            f"with {len(documents)} chunks"
        )

        return vectorstore

    def create_image_vectorstore(
        self,
        image_documents: List[Document]
    ) -> Chroma:

        print("🔄 Creating IMAGE ChromaDB...")

        if self.chroma_image_path.exists():
            shutil.rmtree(self.chroma_image_path)

        vectorstore = Chroma(
            collection_name="disease_image_knowledge",
            persist_directory=str(self.chroma_image_path)
        )

        embeddings = []
        metadatas = []
        documents = []
        ids = []

        for i, doc in enumerate(image_documents):

            embeddings.append(
                doc.metadata["embedding"]
            )

            metadata = doc.metadata.copy()

            del metadata["embedding"]

            metadatas.append(metadata)

            documents.append(doc.page_content)

            ids.append(f"img_{i}")

        vectorstore._collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(
            f"✅ Image vectorstore created "
            f"with {len(image_documents)} images"
        )

        return vectorstore

    ###########################################################################
    # LOAD EXISTING VECTORSTORES
    ###########################################################################

    def load_existing_vectorstores(self):

        text_store = Chroma(
            persist_directory=str(self.chroma_text_path),
            embedding_function=self.text_embeddings,
            collection_name="disease_text_knowledge"
        )

        image_store = Chroma(
            persist_directory=str(self.chroma_image_path),
            collection_name="disease_image_knowledge"
        )

        return text_store, image_store

    ###########################################################################
    # SETUP
    ###########################################################################

    def setup_rag_system(
        self,
        force_rebuild: bool = False
    ):

        print("🚀 Setting up MULTIMODAL RAG...")

        #if (
        #    self.chroma_text_path.exists()
        #    and self.chroma_image_path.exists()
        #    and not force_rebuild
        #):

        #    print("📦 Existing vectorstores found")

        #    return self.load_existing_vectorstores()

        #######################################################################
        # TEXT
        #######################################################################

        #documents = self.load_documents()

        #if not documents:

        #    print("⚠️ No text documents found")

        #    return None, None

        #text_chunks = self.split_documents(documents)

        #text_store = self.create_text_vectorstore(
        #    text_chunks
        #)
        text_store = Chroma(
            persist_directory=str(self.chroma_text_path),
            embedding_function=self.text_embeddings,
            collection_name="disease_text_knowledge"
        )

        #######################################################################
        # IMAGES
        #######################################################################

        image_documents = self.extract_images_from_pdfs()

        image_store = self.create_image_vectorstore(
            image_documents
        )

        print("✅ Multimodal RAG setup completed")

        return text_store, image_store

    ###########################################################################
    # SEARCH
    ###########################################################################

    def search_text(
        self,
        vectorstore: Chroma,
        query: str,
        k: int = 3
    ):

        print(f"\n🔍 TEXT SEARCH: {query}")

        results = vectorstore.similarity_search(
            query,
            k=k
        )

        for i, doc in enumerate(results, 1):

            print(f"\n📄 Text Result {i}")

            print(
                f"File: "
                f"{doc.metadata.get('filename')}"
            )

            print(
                f"Content: "
                f"{doc.page_content[:300]}"
            )

        return results

    def search_images(
        self,
        image_store: Chroma,
        query: str,
        k: int = 3
    ):

        print(f"\n🖼️ IMAGE SEARCH: {query}")

        query_embedding = self.embed_query_image_text(
            query
        )

        results = image_store.similarity_search_by_vector(
            query_embedding,
            k=k
        )

        for i, doc in enumerate(results, 1):

            print(f"\n🖼️ Image Result {i}")

            print(
                f"File: "
                f"{doc.metadata.get('filename')}"
            )

            print(
                f"Page: "
                f"{doc.metadata.get('page')}"
            )

        return results

    ###########################################################################
    # HELPERS
    ###########################################################################

    def _get_doc_type(self, filename: str) -> str:

        if "ficha" in filename.lower():
            return "technical"

        return "general"

    def _generate_doc_id(self, content: str) -> str:

        return hashlib.md5(
            content.encode()
        ).hexdigest()[:8]


###############################################################################
# MAIN
###############################################################################

def main():

    print("🎧 MULTIMODAL RAG - Diseases")
    print("=" * 50)

    processor = DocumentProcessor(
        docs_path=DOCS_PATH
    )

    text_store, image_store = (
        processor.setup_rag_system(
            force_rebuild=True
        )
    )
    #text_store, image_store = (
    #    processor.load_existing_vectorstores()
    #)

    ###########################################################################
    # TESTS
    ###########################################################################

    test_queries = [
        "Fungal disease in leaves",
        "Insect damage symptoms",
        "Brown lesions in agave"
    ]

    for query in test_queries:

        processor.search_text(
            text_store,
            query
        )

        processor.search_images(
            image_store,
            query
        )

    print("\n✅ Setup completed")


if __name__ == "__main__":
    main()