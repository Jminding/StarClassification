from astroquery.simbad import Simbad

# Initialize the Simbad query tool
simbad_query = Simbad()

# Set the spectral class as 'K'
spectral_class = 'K'

# Query for objects with the specified spectral class
result_table = simbad_query.query_criteria(sptype=spectral_class)

# Print the result table
print(result_table)
