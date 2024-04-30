# This is an initial notebook to setup a more automated process to scrape data 
# from nanny's profiles in the GTA.

library(rvest)
library(tidyverse)
library(utils)

#setwd(dirname(rstudioapi::getSourceEditorContext()$path))

main_link <- "https://canadiannanny.ca"
region_link_1 <- "https://canadiannanny.ca/nannies/canada"
file_name <- "profiles_canada"
date <- Sys.Date()
datapath <- paste0("data/input/", date, "/")

# The main site has boxes with the different nannies that are offering their services, 
# on the left there's a menu to filter the profiles according to the services they provide. 
# We can see that the boxes of profiles are under <a> and <div> nodes, 
# and it has a hyperlink which takes you to the main profile site, where we can extract the information.


retrieve_text_div <- function(html_page, class){
  # A general function to retrieve all text under a div node
  text <- html_page |> 
    html_nodes(xpath = paste0("//div[@class = '",class,"']")) |> 
    html_text()
  return(text)
}

profile_links <- function(html_page){
  # Function to retrieve the links to every profile in a page
  links <- html_page |> 
    html_nodes("[class*='sc-dkzDqf koQdGB'] > a") |> 
    html_attr("href")  
  return(links)
}


retrieve_name <- function(html_page){
  # Function to retrieve the name of the Nanny
  name <- html_page |> 
    html_nodes(xpath = "//h2[@data-testid='user.fullName']") |> 
    html_text()
  return(name)
}

retrieve_location <- function(html_page){
  location <- html_page |> 
    html_nodes(xpath = "//div/a[@class='sc-jOrMOR gVaiuT sc-4c1f97a-0 DQVmw']") |> 
    html_text()
  return(location)
}

retrieve_reasons <- function(html_page){
  # The Reasons to Hire Me section had a unique format so we had to create
  # this specific function.
  reasons <- html_page |> 
    html_node(xpath = "//h2[text() = 'Reasons to Hire Me']") |> 
    html_nodes(xpath = "./following-sibling::ul") |> 
    html_children() |> 
    html_text() |> 
    paste(collapse = " ")
  return(reasons)
}


text_subheading_subsection <- function(html_page, subheading, subsection){
  # In this part I'm first finding the subheading of interest, and retrieving 
  # all div nodes that follow.
  # Then I'm selecting the div node with the subsection of interest and retrieving 
  # the next div node.
  # The lists are store in span nodes, so at the end I retrieve those and print text.
  text <- html_page |> 
    html_nodes(xpath = paste0("//h2[text() ='",subheading,"']")) |> 
    html_nodes(xpath = "./following-sibling::div") |>
    html_nodes(xpath = paste0("//div[text() ='",subsection,"']")) |>
    html_nodes(xpath = "./following-sibling::div") |> 
    html_nodes("span") |> 
    html_text() 
  text <- paste(text[nzchar(text)], collapse = ", ")
  return(text)
}


star_rating <- function(html_page){
  # Function to retrieve star rating
  white_stars <- html_page |> 
    html_nodes(xpath = "//div/svg[@data-icon='star']") |> 
    html_children() |> 
    html_attr("d") |> 
    str_starts("M287") |> 
    which() 
  
  if (length(white_stars) == 0){
    return(5)
  }
  
  return(white_stars[1] - 1)
}


main_tibble_builder <- function(links){
  # This function calls all other scraping functions to create a tibble with
  # columns that are specified below. 
  # The list format of R is used to build the tibble, one row at a time. No 
  # way exists as far as I know.
  div_classes <- c("sc-dkzDqf eOkHek", "sc-bczRLJ sc-83be0272-0 hwaJrK dQnxdr",
                   "sc-jSMfEi sc-c85845a3-0 cRFjox lgGVth", 
                   "sc-jOrMOR gVaiuT sc-4c1f97a-0 DQVmw","sc-7dd3eb4d-1 kjcbRe", 
                   "sc-dkzDqf hroDAb", "sc-bczRLJ lhvxZe", "sc-21eb5dfa-1 gJGBuK")
  names(div_classes) <- c("reviews", "short_blurb", "about_me", "location", "pages",
                          "Active", "years_exp", "rate")
  
  num_cols <- 21
  my_list <- vector("list", length  = num_cols)
  names(my_list) <- c("name", "num_reviews", "star_rating", "short_blurb", "location", "reasons",
                      "about_me", "availability_work", "availability_need", "experience_ages",
                      "experience_children", "experience_conditions",
                      "details_transport", "qualifications_provide",
                      "qualifications_languages", "services_responsabilities", "Active",
                      "years_exp", "rate", "link", "date")
  
  columns <- names(my_list)
  for (i in seq_along(links)){
    for (colname in columns[!columns %in% c("date", "link")]){
      my_list[[colname]][i] <- NA
    }
    my_list[["date"]][i] <- as.character(date)
    my_list[["link"]][i] <- links[i]
    my_list[["Active"]][i] <- "Profile was removed"
  }
  
  #env <- environment()
  #env$ans <- rep(TRUE, length(links)) 
  pb = txtProgressBar(min = 0, max = length(links), initial = 0) 
  for (i in seq_along(links)){
    tryCatch({
      profile_page <- read_html(links[i])
      
      #my_list[["link"]][i] <- links[i]
      my_list[["name"]][i] <- retrieve_name(profile_page)
      my_list[["num_reviews"]][i] <- retrieve_text_div(profile_page, div_classes["reviews"])
      my_list[["short_blurb"]][i] <- retrieve_text_div(profile_page, 
                                                       div_classes["short_blurb"])
      about_me_text <- retrieve_text_div(profile_page, 
                                         div_classes["about_me"])
      my_list[["about_me"]][i] <- ifelse(length(about_me_text) == 0, "", about_me_text)
      my_list[["Active"]][i] <- retrieve_text_div(profile_page, div_classes["Active"])
      
      my_list[["location"]][i] <- retrieve_location(profile_page)
      my_list[["reasons"]][i] <- retrieve_reasons(profile_page)
      
      my_list[["availability_work"]][i] <- text_subheading_subsection(profile_page, 
                                                                      "Availability", "I can work:")
      my_list[["availability_need"]][i] <- text_subheading_subsection(profile_page, 
                                                                      "Availability", "I need:")
      my_list[["experience_ages"]][i] <- text_subheading_subsection(profile_page, 
                                                                    "Experience", "Ages include:")
      my_list[["experience_children"]][i] <- text_subheading_subsection(profile_page, 
                                                                        "Experience", "I can look after:")
      my_list[["experience_conditions"]][i] <- text_subheading_subsection(profile_page, 
                                                                          "Experience", 
                                                                          "I have advanced experience with:")
      my_list[["details_transport"]][i] <- text_subheading_subsection(profile_page, 
                                                                      "Job details", "Transportation:")
      my_list[["qualifications_provide"]][i] <- text_subheading_subsection(profile_page, 
                                                                           "Qualifications", "I can provide:")
      my_list[["qualifications_languages"]][i] <- text_subheading_subsection(profile_page, 
                                                                             "Qualifications", "I can speak")
      my_list[["services_responsabilities"]][i] <- text_subheading_subsection(profile_page, 
                                                                              "Services", 
                                                                              "Responsibilities include:")
      
      # In this part of the function we retrieve the rate and years of experience
      # First step is to get the text under the node where the info is written.
      temp_vec <- profile_page |> 
        html_nodes("[class*='sc-bczRLJ sc-aa85deb1-0 YbNpD iElYIb'] > div") |> 
        html_text()
      # In some profiles either rate or experience is missing or both, so we use
      # grepl to see if the text include "exp" (x years exp.) or "hour" ($x/hour).
      exp_test <- grepl("exp", temp_vec)
      rate_test <- grepl("hour", temp_vec)
      my_list[["years_exp"]][i] <- ifelse(any(exp_test), temp_vec[exp_test], NA)   
      my_list[["rate"]][i] <- ifelse(any(rate_test), temp_vec[rate_test], NA)
      
      #my_list[["date"]][i] <- as.character(date)
      
      if (grepl("No reviews yet", my_list[["num_reviews"]][i])){
        my_list[["star_rating"]][i] <- NA
      }else{
        my_list[["star_rating"]] <- star_rating(profile_page)
      }
      
      
    }, error = function(e){
      print(paste0("Errors in profile: ", i, ". Link: ", links[i]))
      
    })
    setTxtProgressBar(pb,i)
  }
  return(as_tibble(my_list))
}

add_star_rating <- function(data){
  # This function is attempting fix an issue with the star ratings. 
  # Initially we tried to retrieved the star ratings directly when doing the scraping.
  # However, it was found that a lot of rating weren't being retrieved correctly.
  # This fix runs that part of the scraping again for those profiles that have reviews.
  
  # Retrived the row numbers of the profiles with reviews
  profiles_w_reviews <- which(!grepl("No reviews yet", data$num_reviews))
  
  # Iterate over them
  for (i in profiles_w_reviews){
    tryCatch({
      profile_page <- read_html(data$link[i])
      data$star_rating[i] <- star_rating(profile_page)
    }, error = function(err){
      print(paste0("Error at retrieving star rating in profile: ", i, 
                   ". url: ", data$link[i]))
    })
  }
  return(data)
}


# First approach: iterate over all pages one by one.
data_downloader <- function(num_pages, region_link = region_link_1, 
                            filename = file_name, path = datapath ){
  # This function iterates over all pages one by one.
  # Its main purpose is to call the main_tibble_builder function and write 
  # the resulting tibble to a csv file.
  start <- Sys.time()
  pages <- 2:num_pages
  pages_link <- paste0(region_link, "?page=")
  links <- c(region_link, paste0(pages_link, pages))
  tibble_list <- list()
  
  if (!file.exists(path)){
    dir.create(path)
  }
  
  for (page_num in seq_along(links)){
    print(paste0("Retrieving profiles from page: ", page_num, "/", num_pages))
    
    site <- read_html(links[page_num])
    profiles <- paste0(main_link, profile_links(site))
    data <- main_tibble_builder(profiles) 
    data <- add_ids(data)
    add_star_rating(data) |> 
      separate(location, c("city", "province"), sep = ",") |> 
      write.csv(paste0(path,"/",filename,"_",page_num,".csv"), row.names = F)
  }
  total_time <- Sys.time() - start
  
  print(paste0("Finished download, total time elapse: ", total_time))
}

add_ids <- function(tibble, path = "data/input/"){
    link_ids <- read.csv(paste0(path, "link_ids.csv"))
    
    missing_links <- tibble |> 
      select(link) |>  
      anti_join(link_ids, join_by(link))
    
    if (dim(missing_links)[1] != 0){
      missing_links_ids <- (max(link_ids$id)+1):(max(link_ids$id)+dim(missing_links)[1])
      missing_links_tibble <- cbind(missing_links_ids, missing_links)
    
      names(missing_links_tibble) <- c("id","link")
      link_ids <- rbind(link_ids, missing_links_tibble)
      link_ids |> write.csv(paste0(path, "link_ids.csv"), row.names = F)
    }
    
    tibble <- tibble |> 
      left_join( link_ids, join_by(link), relationship = "many-to-many") |> 
      select(id, everything())
    
    return(tibble)
}

concat_tibbles <- function(filename = file_name, input_path = datapath,
                           output_path = paste0(datapath, "profiles_canada.csv"), 
                           rm.files = F){
  # This function reads all the .csv files in the directory and puts them in a tibble list  
  # if they have the correct dimensions. At the end we bind the rows of this list
  # and remove duplicated values.
  list_of_files <- list.files(input_path, pattern = filename, 
                              recursive = T, 
                              full.names = T)
  tibble_list <- list()
  tryCatch({
  for (file in list_of_files){
    print(file)
    tibble <- read.csv(file)
    tibble_list[[file]] <- tibble
  }
  bind_rows(tibble_list) |> distinct() |> 
    write.csv(output_path, row.names = F)
  }, error = function(err){
    stop("Probably error at concatenate or writing. Check tibble dimensions and output path.")
  })
  
  if (rm.files){
    file.remove(list_of_files[list_of_files != output_path])
  }
}



default_image <- as.raster(image_read("data/input/image_base.jpeg"))

# Function to download profile pictures
picture_downloader <- function(html_page, filename, resize="30%"){
  profile_pic_src <- html_page |> 
    html_nodes(xpath = "//div[@class = 'sc-83be0272-3 dZmLIg']") |> 
    html_nodes("img") |> 
    html_attr("src")
  
  img <- image_read(paste0(main_link, profile_pic_src)) |> 
    image_resize(resize)
  
  if (!identical(as.raster(img), default_image)){
    img |> image_write(filename) 
  }
}


# ===================================================================== #
# MAIN
# ===================================================================== #

#download <- readline(prompt = "Download profiles? [y/n]: ")
#if (download == "y"){
#  number_pages <-  as.numeric(readline(prompt = "Number of pages to retrieve: "))
#  data_downloader(number_pages)
#  concat_tibbles(output_path = paste0(datapath, "profiles_canada.csv"))
#}else{
#  print("Attaching functions to environment...")
#}