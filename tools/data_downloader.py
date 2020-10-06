from os.path import dirname, abspath
import wget
root_dir = dirname(dirname(abspath(__file__)))

# In case download failed, go to https://gofile.io/d/IKXe81 and download jsons to dataset


def main():
    print(root_dir)
    print("Start to download dataset to ~/dataset/...")

    # Download business.json
    url_1 = 'https://srv-file19.gofile.io/download/IKXe81/business.json'
    wget.download(url_1, "{}/dataset/".format(root_dir))

    # Download reviews.json
    url_2 = 'https://srv-file19.gofile.io/download/IKXe81/reviews.json'
    wget.download(url_2, "{}/dataset/".format(root_dir))

    # Download tips.json
    url_3 = 'https://srv-file19.gofile.io/download/IKXe81/tips.json'
    wget.download(url_3, "{}/dataset/".format(root_dir))

    # Download users.json
    url_4 = 'https://srv-file19.gofile.io/download/IKXe81/users.json'
    wget.download(url_4, "{}/dataset/".format(root_dir))

    print("\nDownload finished.")


if __name__ == "__main__":
    main()