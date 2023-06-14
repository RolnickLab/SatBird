#!/usr/bin/awk -f
#
# GNU AWK SCRIPT
#
# Fields are tab-separated
# Field 27 is LOCALITY ID
# Field 31 is OBSERVATION DATE
# Field 42 is ALL SPECIES REPORTED
#
function lid(l){
  return substr(l, 2) + 0
}
function add_to_buffer(l, d){
  if(l in BUFFER){
    BUFFER[l] = BUFFER[l] d
  }else{
    BUFFER[l] = d
  }
  BUFFER_SIZE += length(d)
}
function flush_buffer(){
  for(l in BUFFER){
    i = lid(l)
    d = sprintf("output/split-%03d/", i % 1000)
    f = d l ".csv"
    
    # Create parent directory if not yet created
    if(!BUFFERDIRS[d]){
      BUFFERDIRS[d] = 1
      system("mkdir -p " d);
    }
    
    # Print TSV header if this is the first write to the file
    if(LOCALITIES[l]!="1"){
      LOCALITIES[l]="1"
      print TSVHEADER >> f
    }
    
    # Write out buffered data for this file, then empty buffer and close the file.
    printf "%s", BUFFER[l] >> f
    delete       BUFFER[l]
    close(f);
  }
  BUFFER_SIZE=0
}
BEGIN{
  FS = "\t"
  getline
  TSVHEADER = $0
  split("", LOCALITIES)
  split("", BUFFERDIRS)
  split("", BUFFER)
  BUFFER_SIZE=0
}
{
  if($42=="1" && match($31, "20[12][0-9]-(01|06|07|12)-*")){
    add_to_buffer($27, $0 ORS)
    if(BUFFER_SIZE >= 128*1024*1024){
      flush_buffer()
    }
  }
}
END{
  flush_buffer()
}

