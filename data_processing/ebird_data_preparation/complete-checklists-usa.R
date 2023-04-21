library(auk)
library(dplyr)
# Goal: Extract complete checklists, only retain needed columns. matt (from Ebird repo) gave this code

# Change file path to /miniscratch/srishtiy/ebd_sampling_relApr-2021.txt
f_filtered <- auk_sampling("/home/srishtiy/Downloads/eBird-Notebook/sample_event_data_april_21/ebd_sampling_relApr-2021.txt") %>% 
  auk_complete() %>%
  auk_country(country = "United States") %>%
  auk_filter("complete-checklists_usa.txt", overwrite = TRUE,
             keep = c("state", "country", "latitude", "longitude",
                      "group identifier", "sampling event identifier",
                      "locality", "locality id", "locality type",
                      "observer_id"))

# read into r as a data frame
checklists <- read_sampling(f_filtered)

# number of complete checklists in USA
n_checklists_usa <- checklists %>%
  # only keep checklists at hotspots
  filter(country == "H") %>%
  count(locality_id, locality)

write.csv(n_checklists_usa, "n_checklists_usa.csv", row.names = FALSE)
