import requests
import os
from tqdm import tqdm
from bs4 import BeautifulSoup as bs
from urllib.parse import urljoin, urlparse


def is_valid(url: str) -> bool:
    """
    Проверяем, является ли url действительным URL
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def get_all_images(url: str) -> [str]:
    """
    Возвращает все URL‑адреса изображений по одному `url`
    """
    soup = bs(requests.get(url).content, "html.parser")
    urls = []
    for img in tqdm(soup.find_all("img"), "Получено изображение"):
        img_url = img.attrs.get("src")
        if not img_url:
            # если img не содержит атрибута src, просто пропускаем
            continue
        # сделаем URL абсолютным, присоединив имя домена к только что извлеченному URL
        img_url = urljoin(url, img_url)
        # удалим URL‑адреса типа '/hsts-pixel.gif?c=4.3.5'
        try:
            pos = img_url.index("?")
            img_url = img_url[:pos]
        except ValueError:
            pass
        # наконец, если URL действителен
        if is_valid(img_url):
            urls.append(img_url)
    return urls


def download(url: str, pathname: str) -> None:
    """
    Загружает файл по URL‑адресу и помещает его в папку `pathname`
    """
    if not os.path.isdir(pathname):
        os.makedirs(pathname)
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("Content-Length", 0))
    filename_png = os.path.splitext(url.split("/")[-1])[0] + ".png"
    filename = os.path.join(pathname, filename_png)
    progress = tqdm(response.iter_content(1024), f"Загружен {filename}", total=file_size, unit="B", unit_scale=True,
                    unit_divisor=1024)
    with open(filename, "wb") as f:
        for data in progress.iterable:
            f.write(data)
            progress.update(len(data))


def main(url: str, path: str) -> None:
    imgs = get_all_images(url)
    for img in imgs:
        download(img, path)


if __name__ == "__main__":
    url = 'https://publicdomainvectors.org/en/free-clipart/signs-symbols/'
    path = 'data/pictures/1'
    main(url, path)
