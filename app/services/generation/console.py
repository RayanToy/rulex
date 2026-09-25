"""Вывод хода генерации в консоль."""
import sys


def log(message: str) -> None:
    """Печать, устойчивая к кодировке консоли.

    Сообщения об ошибках приходят из API и могут содержать символы,
    которых нет в cp1251 (например, полноширинный ＄). Обычный print()
    на такой строке падает с UnicodeEncodeError — то есть обработчик
    ошибки сам роняет процесс.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    sys.stdout.write(message.encode(encoding, errors="replace").decode(encoding) + "\n")
