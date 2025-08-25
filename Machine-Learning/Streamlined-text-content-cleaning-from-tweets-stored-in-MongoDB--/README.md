# __<center>Streamlined text content cleaning from tweets stored in MongoDB</center>__

## __<center>Overview</center>__
Using MapReduce paradigms to process and analyze a large movie dataset, revealing trends in movie production by year and company. The workflow incorporates MongoDB for data storage and mrjob for MapReduce implementation, enabling scalable data processing and sorting.


## __<center>Dataset</center>__
- [`movies.zip`](https://github.com/VivianNg9/Data-Mining/blob/main/MapReduce/movies.json): Contains the movie dataset used for analysis. It includes data such as release dates and production companies.
  
![MongoDB](https://github.com/VivianNg9/Streamlined-text-content-cleaning-from-tweets-stored-in-MongoDB--/blob/main/image%20/MongoDB.png)


## __<center>Project Environment</center>__
- **MongoDB & Studio 3T:** Used for database creation and dataset importation.
- **pymongo:** To connect to MongoDB and extract document data.
- **mrjob:** Framework for implementing MapReduce programs in Python.


## __<center>Project Workflow</center>__

### Task 1: Movie Count by Production Company and Year  
Calculate the number of movies released by each production company for each year.
  
![Workflow1](https://github.com/VivianNg9/Streamlined-text-content-cleaning-from-tweets-stored-in-MongoDB--/blob/main/image%20/workflow1.png)

**EXTRACTION:** 

Using pymongo to extract <year, company> for each movie.
  1. **Data Retrieval:** Access 'movie -> date' to get release dates and extract the release year.
  2. **Company Identification:** Identify the top three production companies from 'movie -> companies'.
  3. **Data Formatting:** Create `<year, company>` pairs.
  4. **Data Storage:** Store these pairs in [`year_and_company.txt`](https://github.com/VivianNg9/Data-Mining/blob/main/MapReduce/Output%20file%20for%20Task%201/year_and_company.txt).
     
```python
from pymongo import MongoClient                                                                                                   

# Making database connection
client = MongoClient('localhost', 27017)
db = client["assignment1"]
col = db["movies"]
cur = col.find() 

# Write a txt file with <year, company> pairs of all movies 
file = open("year_and_company.txt", "w")
# Loop through each movie document in the cur
for movie in cur: 
    for i in range(min(len(movie['companies']), 3)):
        year_company = movie['date'][-4:] + " , " + movie['companies'][i]['name']+ '\n'
    file.write(year_company)  
file.close()
```

**COUNT:** 

Use mrjob to calculate the frequency of each `<year, company>` pair. Output stored in [`task1_output.txt`](https://github.com/VivianNg9/Data-Mining/blob/main/MapReduce/Output%20file%20for%20Task%201/task1_output.txt)

```python
from mrjob.job import MRJob                                     

class MRWordCount(MRJob):
    def mapper(self, _, line):
        yield line, 1

    def reducer(self, year_company, count):
        yield year_company, sum(count)

if __name__ == "__main__":
    MRWordCount.run()
```


### Task 2: Sorting Algorithms Implementation 
Implement the Merge Sort1 algorithm and the Bucket Sort2 algorithm using two MapReduce programs, to sort results of Task 1.

**MERGE SORT:** 

Sort `<year, company>` pairs in ascending order. 

![Merge sort](https://github.com/VivianNg9/Streamlined-text-content-cleaning-from-tweets-stored-in-MongoDB--/blob/main/image%20/Workflow_Mergesort.png)

Output stored in [`task2_mergesort.txt`](https://github.com/VivianNg9/Data-Mining/blob/main/MapReduce/Output%20file%20for%20Task%202/task2_mergesort_output.txt)

```python
from mrjob.job import MRJob                                                           
from mrjob.step import MRStep                                                        

class MRMergeSort(MRJob):

    def steps(self):
        return [
            MRStep(mapper=self.mapper, reducer=self.reducer)
        ]

    def mapper(self, _, line):
        year_company, count = line.split('\t')
        count = int(count)
         # "key" for combining year_company and count 
        yield "key", (year_company[1:-1], count) 

    def merge_sort(self, arr):
        if len(arr) > 1:
            mid = len(arr) // 2
            left_half = arr[:mid]
            right_half = arr[mid:]

            self.merge_sort(left_half)
            self.merge_sort(right_half)

            i = j = k = 0
            while i < len(left_half) and j < len(right_half):
                # Sort in ascending order
                if left_half[i][1] < right_half[j][1]:  
                    arr[k] = left_half[i]
                    i += 1
                else:
                    arr[k] = right_half[j]
                    j += 1
                k += 1

            while i < len(left_half):
                arr[k] = left_half[i]
                i += 1
                k += 1

            while j < len(right_half):
                arr[k] = right_half[j]
                j += 1
                k += 1

        # Return the sorted array
        return arr  

    def reducer(self, _, tuples):
        sorted_tuples = self.merge_sort(list(tuples))
        for tuple in sorted_tuples:
            yield tuple[0], tuple[1]

if __name__ == '__main__':
    MRMergeSort.run()
```

**BUCKET SORT:** 

Sort `<year, company>` pairs in descending order.

![Bucket sort](https://github.com/VivianNg9/Streamlined-text-content-cleaning-from-tweets-stored-in-MongoDB--/blob/main/image%20/Workflow_Bucketsort.png)

Output stored in [`task2_bucketsort_output.txt`](https://github.com/VivianNg9/Data-Mining/blob/main/MapReduce/Output%20file%20for%20Task%202/task2_bucketsort_output.txt)

``` python
from mrjob.job import MRJob                                                                     
from mrjob.step import MRStep                                                                    

class MRBucketSort(MRJob):
 
    def configure_args(self):
        super(MRBucketSort, self).configure_args()
        self.add_passthru_arg('--num_buckets', type=int, default=250)
        self.add_passthru_arg('--bucket_size', type=int, default=3)
 
    def steps(self):
        return [
            MRStep(
                mapper=self.bucket_assignment_mapper
            ),
            MRStep(
                reducer=self.bucket_sort_reducer
            ),
            MRStep(
                reducer=self.bucketid_sort_reducer
            )
        ]

    def bucket_assignment_mapper(self, _, line):
        year_company, count = line.split('\t')
        count = int(count)
        bucket_id = count // self.options.bucket_size
        yield bucket_id, (year_company[1:-1], count)


    def bucket_sort_reducer(self, bucket_id, records):
        # Sort in descending order
        sorted_records = sorted(records, key=lambda x: (-x[1], x[0]))  
        for record in sorted_records:
            yield "key",(bucket_id, record)
    
    def bucketid_sort_reducer(self, key, bucketid_records):
        for value in sorted(bucketid_records, key=lambda x:x[0], reverse=True):
            yield value[1]

if __name__ == '__main__':
    MRBucketSort.run()
```


