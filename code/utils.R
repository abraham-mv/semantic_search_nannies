# Initial code to create indices for the profiles.


library(tidyverse)
source("code/scraping.R")

input_path = "data/input/"

#list.files(input_path, pattern = "profiles_")


create_ids <- function(input_path = "data/input/"){
  # Function to create a unique identifier for profiles based on their url.
  # It was mean to only be ran once.
  
  file_list_lists <- list()
  for (folder in list.dirs(input_path)){
      file_list_lists[[folder]] <- list.files(folder, pattern = "profiles_", full.names = T)
    
  }
  
  file_list <- unlist(file_list_lists, recursive=T)
  names(file_list) <- NULL
  
  
  links_list <- list()
  for (i in seq_along(file_list)){
    links_list[[i]] <- read.csv(file_list[i]) |> select(link)
  }
  unique_links <- bind_rows(links_list) |> unique()
  id <- 1:dim(unique_links)[1]
  links_with_ids <- cbind(id, unique_links)
  links_with_ids |> write.csv(paste0(input_path, "link_ids.csv"))
}




get_ratings <- function(profiles){
  # Function to extract star ratings.

  #profiles <- read.csv("data/input/nannies_profiles_canada.csv") 
  profiles_w_reviews <- which(!grepl("No reviews yet", profiles$num_reviews))
  
  
  for (i in profiles_w_reviews){
    tryCatch({
      profile_page <- read_html(profiles$link[i])
      profiles$star_rating[i] <- star_rating(profile_page)
    }, error = function(err){
      print(paste0("Error in profile: ", i, ". url: ", profiles$link[i]))
    })
  }
  
  #profiles |> filter(!grepl("No reviews yet", profiles$num_reviews)) |> 
  #  select(num_reviews, star_rating)
  
  #profiles |> write.csv("data/input/nannies_profiles_canada.csv", row.names = F)
  return(profiles)
}





get_new_profiles <- function(n_dates){
  # Retrieves profiles that haven't been retrieved in the last n_dates.
  # Input: 
  #       n_dates: integer number of dates.   
  # Output:
  #       csv file with the scraped profiles.
  
  dir.create("data/input/temp")
  profiles <- read.csv("data/input/nannies_profiles_canada.csv") |> 
    mutate(rate = as.numeric(str_extract(rate, "\\d+(?:\\.\\d+)?")), 
           years_exp = as.numeric(str_extract(years_exp, "\\d+(?:\\.\\d+)?")), 
           date = as.Date(date)) 
  
  old_links <- profiles |> 
    distinct(date, link) |> 
    mutate(date = as.character(date)) |> 
    group_by(link) |> 
    summarise(dates = paste0(date, collapse = " ")) |> 
    filter(!grepl(paste0(tail(sort(unique(profiles$date)), n_dates), collapse = "|"), dates)) |> 
    select(link) |> pull()
  
  
  for (i in seq(12, length(old_links), 12) ){
    tryCatch({
      current_links <- old_links[(i-11):i]
      profiles_temp <- main_tibble_builder(current_links)
      profiles_temp |> write.csv(paste0("data/input/temp/profiles_new-",i, ".csv"), 
                                 row.names = F)
    }, error = function(e){})
  }
  
  list.files("data/input/temp", pattern = "profiles_new", full.names = T)
  
  concat_tibbles("profiles_new", "data/input/temp/", 
                 output_path = "data/input/nannies_profiles_new.csv")
}