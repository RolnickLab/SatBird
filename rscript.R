library('ebirdst')
library('dplyr')
library('sf')
#set_ebirdst_access_key("k27o1qsob07u")
datalist = list()
df=read.csv(file = 'species_data.csv')
j<-1
for (s in unique(df$scientific_name)) {
  
  # download the example data
  #s<-ebirdst_runs[i,2][[1]]
  print(s)
  #print(i)
  ebirdst_download(species =s)
  #path='/network/scratch/a/amna.elmustafa/tmp2/ecosystem-embedding')
  
  # get the path
  path <- get_species_path(s)
  
  # use it to load data
  abd <- load_ranges(path = path, resolution = 'lr')
  
  # filter the season     
  # if('resident' %in% unique(abd$season)){
  #  abd<-filter(abd, season=="resident")
  # geom<- abd["geom"]
  #}
  
  #  else{
  i<-1
  geoms=list()
  for(seas in unique(abd$season)){
    print(seas)
    #print(class(seas))
    x<-filter(abd,season==seas)
    if ((format(x['start_date'][[1]],"%m")<="06")&& format(x['end_date'][[1]],"%m")>="06"){
      if (i==1){
        geom<-abd['geom']
        geoms[[i]]<-geom
      }
      else{
        #union geoms
        geoms[[i]]<-abd['geom']
        #geom<-st_union(geom,abd['geom'])
        
      }
    }
    i<-i+1
    dt<-do.call(rbind, geoms)
  }
  #casting
  #geom<- sf::st_cast(geom, "MULTIPOLYGON")
  
  
  #if(format(abd['breeding_start'],"%m")  =="06" ||format(abd['breeding_end'],"%m")  =="06"){
  
  #}
  #append to previous dataframe 
  print(class(dt))
  dt$scientific_name<-s
  #print(dt)
  
  datalist[[j]]<-dt
  
  j<-j+1
  #print(do.call(rbind, datalist))
}
#big_data = do.call(rbind, datalist)
big_data<-dplyr::bind_rows(datalist)
# print(class(big_data))
# print(typeof(big_data))
st_write(big_data,'bigdata.shp')
