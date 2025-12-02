# upload this script to colab or write this in a code cell

from google.colab import drive
drive.mount('/content/drive')

# colab will open a prompt and ask to validate drive credentials
# content is accessible in /content/drive

'''
what I then do is
cd /home ; mkdir anupam ; cd anupam ; cp -r /content/drive/MyDrive/<path-to-my-tar.gz> .
the tar xvzf <.tar.gz> : this inflates the code and dataset and I can use GPU training
'''



