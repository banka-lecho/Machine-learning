from save_read import read_files, save_data
from feature_extracting.vit import get_vit_embs
import pandas as pd


def run_vit(num: int) -> None:
    dict_emb = get_vit_embs(f'./data/pictures/{num}')
    save_data(dict_emb, f'./data/embs/embs_targets_{num}.pkl')


def create_dataset(df_name: str) -> None:
    dict_emb = read_files(f'./data/embs/embs_targets_{1}.pkl')
    data = [(path, 1) for path, _ in dict_emb.items()]
    df = pd.DataFrame(data, columns=['Path', 'Class'])

    dict_emb = read_files(f'./data/embs/embs_targets_{2}.pkl')
    data = [(path, 2) for path, _ in dict_emb.items()]
    new_df = pd.DataFrame(data, columns=['Path', 'Class'])
    df = pd.concat([df, new_df], ignore_index=True)

    dict_emb = read_files(f'./data/embs/embs_targets_{3}.pkl')
    data = [(path, 3) for path, _ in dict_emb.items()]
    new_df = pd.DataFrame(data, columns=['Path', 'Class'])
    df = pd.concat([df, new_df], ignore_index=True)

    df = df.set_index('Path')
    df.to_csv(df_name)


def merge_dicts():
    dict_emb1 = read_files(f'./data/embs/embs_targets_{1}.pkl')
    dict_emb2 = read_files(f'./data/embs/embs_targets_{2}.pkl')
    dict_emb3 = read_files(f'./data/embs/embs_targets_{3}.pkl')
    dict_emb1.update(dict_emb2)
    dict_emb1.update(dict_emb3)
    save_data(dict_emb1, f'./data/embs/all_embs.pkl')


if __name__ == "__main__":
    df_name = "./data/csvs/binary.csv"
    # запускаем vit
    run_vit(1)
    run_vit(2)
    run_vit(3)
    # создаём датасет
    create_dataset(df_name)
    merge_dicts()
