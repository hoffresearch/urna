# what is a .urna file

a `.urna` file is a single, self-contained semantic knowledge base. one file
carries the text chunks, their embeddings, the byte spans back to the original
source, optional approximate-nearest-neighbor and lexical indices, and a
search contract that says how the file is meant to be queried.

you copy a `.urna` the way you copy a sqlite database: there are no companion
files, no schema migration, and no external service to look up. the binary
format is open and frozen at version one, so a file written today still opens
years from now.

the runtime memory-maps the file and scores embeddings directly off disk with
simd dot products. there is no server and no central index. the file is the
database.
