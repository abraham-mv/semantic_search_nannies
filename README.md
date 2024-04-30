# Analyzing Text Descriptions

## Scraping
Text descriptions, profie pictures and other variables, such as, rate and years of experience, were scraped from the site [CanadianNanny](https://canadiannanny.ca/). 
To start scraping profiles clone repository to local and run in the R console from the main directory:
``` r
source("code/scraping.R")
data_downloader(num_pages)
concat_tibbles(output_path = "you_path", rm.files=T)
```
For the image classification task we download profile pictures of only those users whose ages have been retrieved. This information is stored in the user_ages.csv file.
``` r
source("code/scraping.R")
main_link <- "https://canadiannanny.ca"
input_path <- "../data/input/"
filename <- "data/output/python_tests/user_ages.csv"
tibble <- read.csv(filename) |> distinct(id, .keep_all = T)
output_path <- "data/input/images/"
# Download profile pictures
for (i in seq_along(tibble$link)){
  tryCatch({
    profile_page <- read_html(tibble$link[i])
    picture_downloader(profile_page, 
                       paste0(output_path,tibble$id[i],".jpeg"))
    }, error = function(err) {
    print(paste0("Error in profile: ",tibble$id[i],". Name: ",tibble$name[i],
                 ". Link: ", tibble$link[i]))
  })
}
```

## Calculating text embeddings
To take advantage of the GPU I suggest computing text embeddings on Google Colab. Any model supported by [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) or HuggingFace is valid. Empiracally I've seen that multi-qa-mpnet-base-cos-v1 gives the best results but feel free to experiment with simpler and more complex models. Up to this point I haven't done any fine tuning, which could be interesting to explore in future iterations of the project. 
``` python
# Mount colab notebook to google drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repo to drive
!git clone https://github.com/abraham-mv/semantic_search_nannies.git
!pip install -q -U sentence-transformers rank_bm25

# Go to the repo folder and pull latest changes
%cd /content/nanny-scraping/
!git pull
%cd code

# Download this otherwise it breaks down
import nltk
nltk.download("punkt")

# Import the model
from nannies_encoder_search import InformationRetrieval
nannies_ir = InformationRetrieval('multi-qa-mpnet-base-cos-v1', save_embeddings=True)
nannies_ir.sentence_bi_encoder()
nannies_ir.embeddings_saver()
```
The embeddings will appear on the folder: data/output/python_tests/model_name/embeddings_date.pkl on the left-hand side of the screen in colab, download this file to your working directory in your local machine (this might take a few minutes). Then you can proceed to the next part. 

## Information Retrieval
We use both the text embeddings computed by the transformer model and another lexical search algorithm called BM25, to extract mainly age and country of origin for all users who mentioned this in their descriptions. If the age or birth country of a certain user has already been retrieved we skip it and only manually check new ones. Check the running_semantic_search notebook for more detail information on the workflow of extracting information. 
