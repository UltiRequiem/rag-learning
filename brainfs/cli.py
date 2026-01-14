"""Command-line interface for Brainfs."""

import builtins
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .database import Document, DocumentDatabase, calculate_file_hash
from .exceptions import BrainfsError
from .llm import LLMClient
from .parsers import DocumentParserFactory, parse_document
from .text import smart_chunk
from .tokenizer import Tokenizer
from .vector_store import VectorStore

console = Console()


@click.group()
@click.version_option()
def main():
    """Brainfs - A CLI tool for document indexing and querying using vector search."""
    pass


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--chunk-method", default="sentences", help="Chunking method: sentences, paragraphs, words"
)
@click.option("--chunk-size", default=3, help="Sentences per chunk (for sentence method)")
@click.option("--chunk-overlap", default=1, help="Overlap between chunks")
@click.option("--recursive", "-r", is_flag=True, help="Index files recursively")
def index(path: Path, chunk_method: str, chunk_size: int, chunk_overlap: int, recursive: bool):
    """Index documents from a file or directory."""
    try:
        db = DocumentDatabase()
        files_to_process = _get_files_to_process(path, recursive)

        if not files_to_process:
            console.print("No supported files found.", style="yellow")
            return

        tokenizer = Tokenizer()

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Analyzing documents...", total=len(files_to_process))
            all_chunks = []

            for file_path in files_to_process:
                progress.update(task, description=f"Analyzing {file_path.name}...")
                try:
                    content = parse_document(file_path)
                    chunks = smart_chunk(
                        content,
                        method=chunk_method,
                        sentences_per_chunk=chunk_size,
                        overlap=chunk_overlap,
                        chunk_size=chunk_size,
                    )
                    all_chunks.extend(chunks)
                except Exception as e:
                    console.print(f"Error processing {file_path}: {e}", style="red")
                    continue
                progress.advance(task)

            progress.update(task, description="Training tokenizer...")
            tokenizer.fit(all_chunks)

            task = progress.add_task("Indexing documents...", total=len(files_to_process))
            indexed_count = 0
            skipped_count = 0

            for file_path in files_to_process:
                progress.update(task, description=f"Indexing {file_path.name}...")

                try:
                    file_hash = calculate_file_hash(file_path)

                    if db.document_exists(file_hash):
                        console.print(f"Skipping {file_path.name} (already indexed)", style="dim")
                        skipped_count += 1
                        progress.advance(task)
                        continue

                    # Parse and chunk document
                    content = parse_document(file_path)
                    chunks = smart_chunk(
                        content,
                        method=chunk_method,
                        sentences_per_chunk=chunk_size,
                        overlap=chunk_overlap,
                        chunk_size=chunk_size,  # For word-based chunking
                    )

                    if not chunks:
                        console.print(f"No content extracted from {file_path.name}", style="yellow")
                        progress.advance(task)
                        continue

                    # Vectorize chunks
                    vectors = [tokenizer.embed(chunk) for chunk in chunks]

                    # Create document record
                    doc = Document(
                        id=file_hash[:16],  # Use first 16 chars of hash as ID
                        path=str(file_path.absolute()),
                        filename=file_path.name,
                        content=content[:1000]
                        + ("..." if len(content) > 1000 else ""),  # Store preview
                        file_hash=file_hash,
                        chunks=chunks,
                        vectors=vectors,
                    )

                    # Save to database
                    db.add_document(doc)
                    indexed_count += 1

                except Exception as e:
                    console.print(f"Error indexing {file_path}: {e}", style="red")

                progress.advance(task)

        console.print(
            Panel(
                f"Indexing complete!\n\n"
                f"• [green]{indexed_count}[/green] documents indexed\n"
                f"• [yellow]{skipped_count}[/yellow] documents skipped (already indexed)\n"
                f"• Vocabulary size: [cyan]{len(tokenizer.vocab)}[/cyan] unique words",
                title="Results",
            )
        )

    except Exception as e:
        raise BrainfsError(f"Indexing failed: {e}") from e


@main.command()
@click.argument("query", required=False)
@click.option("--top-k", default=3, help="Number of top results to return")
@click.option("--interactive", "-i", is_flag=True, help="Enter interactive query mode")
@click.option("--generate", "-g", is_flag=True, help="Generate answer using LLM")
def query(query: str, top_k: int, interactive: bool, generate: bool):
    """Query indexed documents."""
    try:
        db = DocumentDatabase()

        docs = db.list_documents()

        if not docs:
            console.print("No documents indexed. Run 'brainfs index <path>' first.", style="red")
            return

        chunk_data = db.search_chunks()

        if not chunk_data:
            console.print("No searchable content found.", style="red")
            return

        # Build vector store
        with console.status("Loading vector index..."):
            # Collect all chunks for tokenizer training
            all_chunks = [chunk[2] for chunk in chunk_data]  # chunk text
            tokenizer = Tokenizer()
            tokenizer.fit(all_chunks)

            # Build vector store
            store = VectorStore()
            for _doc_id, filename, chunk_text, vector in chunk_data:
                # Re-vectorize with current tokenizer (ensures consistency)
                vector = tokenizer.embed(chunk_text)
                store.add_item(f"{filename}: {chunk_text}", vector)

        # Initialize LLM client if needed
        llm_client = None
        if generate:
            try:
                llm_client = LLMClient()
            except Exception as e:
                console.print(f"Warning: LLM not available: {e}", style="yellow")
                generate = False

        if interactive:
            _interactive_query_mode(store, tokenizer, top_k, generate, llm_client)
        else:
            if not query:
                console.print("Please provide a query or use --interactive mode.", style="red")
                return
            _execute_query(store, tokenizer, query, top_k, generate, llm_client)

    except Exception as e:
        raise BrainfsError(f"Query failed: {e}") from e


def _interactive_query_mode(
    store: VectorStore, tokenizer: Tokenizer, top_k: int, generate: bool, llm_client
):
    """Interactive query mode."""
    mode_text = "with LLM answer generation" if generate else "vector search only"
    console.print(
        Panel(
            f"Interactive Query Mode ({mode_text})\n\n"
            "Enter your queries below. Type 'quit' or 'exit' to stop.\n"
            "Use Ctrl+C to interrupt.",
            title="Brainfs Interactive Mode",
        )
    )

    while True:
        try:
            user_query = console.input("\n[bold blue]Query:[/bold blue] ")
            if user_query.lower() in ("quit", "exit", "q"):
                console.print("Goodbye!", style="green")
                break

            if user_query.strip():
                _execute_query(store, tokenizer, user_query, top_k, generate, llm_client)
        except (KeyboardInterrupt, EOFError):
            console.print("\nGoodbye!", style="green")
            break


def _execute_query(
    store: VectorStore, tokenizer: Tokenizer, query: str, top_k: int, generate: bool, llm_client
):
    """Execute a single query."""
    query_vector = tokenizer.embed(query)

    # Check if query vector is empty (all zeros)
    if not any(query_vector):
        console.print(
            "No matching words found in vocabulary. Try using different terms or check if documents are indexed.",
            style="yellow",
        )
        return

    results = store.search(query_vector, top_k=top_k)

    if not results:
        console.print("No results found.", style="yellow")
        return

    console.print(f"\n[bold]Top {len(results)} results for:[/bold] {query}")

    # Extract contexts for LLM
    contexts = []
    for i, (score, context) in enumerate(results, 1):
        # Split context to get filename and text
        if ": " in context:
            filename, text = context.split(": ", 1)
        else:
            filename, text = "Unknown", context

        contexts.append(text)

        console.print(
            Panel(
                f"[bold]File:[/bold] {filename}\n[bold]Score:[/bold] {score:.4f}\n\n{text}",
                title=f"Result {i}",
            )
        )

    # Generate LLM answer if requested
    if generate and llm_client and contexts:
        console.print("\n[bold cyan]Generating answer...[/bold cyan]")
        try:
            answer = llm_client.generate_answer(query, contexts)
            console.print(Panel(answer, title="Generated Answer", border_style="cyan"))
        except Exception as e:
            console.print(f"Error generating answer: {e}", style="red")


@main.command()
def list():
    """List all indexed documents."""
    try:
        db = DocumentDatabase()
        docs = db.list_documents()

        if not docs:
            console.print("No documents indexed.", style="yellow")
            return

        table = Table(title="Indexed Documents")
        table.add_column("ID", style="cyan")
        table.add_column("Filename", style="green")
        table.add_column("Path", style="dim")

        for doc_id, filename, path in docs:
            table.add_row(doc_id, filename, path)

        console.print(table)
        console.print(f"\nTotal: [bold]{len(docs)}[/bold] documents")

    except Exception as e:
        raise BrainfsError(f"Failed to list documents: {e}") from e


@main.command()
@click.confirmation_option(prompt="Are you sure you want to clear all indexed documents?")
def clear():
    """Clear all indexed documents."""
    try:
        db = DocumentDatabase()
        count = db.clear_all()
        console.print(f"Cleared [bold]{count}[/bold] documents from index.", style="green")
    except Exception as e:
        raise BrainfsError(f"Failed to clear index: {e}") from e


@main.command()
def config():
    """Show configuration information."""
    db = DocumentDatabase()
    supported_formats = DocumentParserFactory.supported_extensions()

    console.print(
        Panel(
            f"[bold]Brainfs Configuration[/bold]\n\n"
            f"Database: {db.db_path}\n"
            f"Supported formats: {', '.join(supported_formats)}\n"
            f"Version: {__import__('brainfs').__version__}",
            title="Configuration",
        )
    )


def _get_files_to_process(path: Path, recursive: bool) -> builtins.list[Path]:
    """Get list of files to process."""
    files = []

    if path.is_file():
        if DocumentParserFactory.is_supported(path):
            files.append(path)
        else:
            console.print(f"Unsupported file type: {path.suffix}", style="red")
    else:
        pattern = "**/*" if recursive else "*"
        for file_path in path.glob(pattern):
            if file_path.is_file() and DocumentParserFactory.is_supported(file_path):
                files.append(file_path)

    return files


if __name__ == "__main__":
    try:
        main()
    except BrainfsError as e:
        console.print(f"Error: {e}", style="red")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\nOperation cancelled.", style="yellow")
        sys.exit(1)
