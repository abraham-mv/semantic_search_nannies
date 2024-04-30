library(tidyverse)
library(rvest)
library(magick)

main_link <- "https://canadiannanny.ca"
input_path <- "../data/input/"

default_image <- as.raster(image_read("data/input/image_base.jpeg"))

# Function to dowload profile pictures
picture_downloader <- function(html_page, filename){
  profile_pic_src <- html_page |> 
    html_nodes(xpath = "//div[@class = 'sc-83be0272-3 dZmLIg']") |> 
    html_nodes("img") |> 
    html_attr("src")
  
  img <- image_read(paste0(main_link, profile_pic_src)) |> 
    image_resize("30%")
  
  if (identical(as.raster(img), default_image)){
    img |> image_write(filename) 
  }
}

#data_folders <- list.dirs(input_path, full.names = F, recursive  = F)
#most_recent <- as.character(max(na.omit(as.Date(data_folders))))
#filename <- list.files(paste0(input_path, most_recent), full.names = T)[1]

#images_filepath <- paste0("../../images/",most_recent)
#if (!file.exists(images_filepath)){
#  dir.create(images_filepath)
#}


#filename <- "data/input/nannies_profiles_canada.csv"
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





#for (path in list.files("data/input/images/", full.names = T)){
#  img <- as.raster(image_read(path))
#  if (identical(img, default_image)){
#    print(path)
#    file.remove(path)
#  }
#}

