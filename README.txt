Name- Sai Nadkarni
UIN- 672756678 (snadka2)

Usage:-
Run main.py or main.ipynb, the program will ask for search string as input. Provide a query and the system will fetch 10 ranked webpages, with a prompt to retrieve more if the user requires. Any prompt apart from "y" or "yes" will stop further retrieval. For every prompt, 10 more results will be fetched.

Working:-
The crawled and cleaned webpages are already stored in a pickle file. The main.py/main.ipynb file loads the required pickle files and runs vectorizer operations on it along with the query after issuance.
