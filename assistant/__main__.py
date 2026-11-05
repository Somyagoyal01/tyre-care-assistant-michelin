"""CLI entry point: python -m assistant "your question"."""
import logging
import sys

from .agent import TyreAssistant


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        stream=sys.stderr)
    question = " ".join(sys.argv[1:]) or input("Question: ")
    r = TyreAssistant().ask(question)
    print(f"[route: {r['route']}]\n{r['answer']}")


if __name__ == "__main__":
    main()
