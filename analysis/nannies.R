library(tidyverse)
library(haven)

# d <- read_dta("~/Downloads/ipumsi_00034.dta")
# 
# possible_nannies <- d |> 
#   filter(occ==35) |> 
#   filter(sex==2) |> 
#   filter(indgen>=112) 
# 
# write_csv(possible_nannies, "nannies_census.csv")

# by province

possible_nannies <- read_csv("/nannies_census.csv")

possible_nannies |> 
  group_by(geo1_ca2011) |> 
  tally() |> 
  mutate(prop = n/sum(n)) 

# proportion not citizens

possible_nannies |> 
  group_by(citizen, geo1_ca2011) |> 
  tally() |> 
  group_by(geo1_ca2011) |> 
  summarize(prop_not_citizen = sum(n[citizen==4])/(sum(n[citizen==2]+sum(n[citizen==3]))))

# birth country

possible_nannies |> 
  group_by(bplcountry) |> 
  tally() |> 
  arrange(-n)

# Age distribution

possible_nannies |> 
  filter(empstat==2)  |> 
  ggplot(aes(age)) + 
  geom_histogram()
