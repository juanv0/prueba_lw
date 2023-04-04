import pandas as pd
import mysql.connector as connector
import os


def populate_data():

    os.makedirs('mls_data', exist_ok=True)
    csv_data=os.listdir('mls_data/')

    def get_columns(mls_id, listing_table):

        mls_db = 'mls_%s' %mls_id
        mls_query = "SHOW COLUMNS FROM %s" %listing_table
        file_name = '%s.csv'%mls_db
        path_name = 'mls_data/'+file_name

        try:
            print('working on: %s'%mls_db)
            if(file_name in csv_data):
                return pd.read_csv(path_name)
            
            with connector.connect(user='juanvasquez', password='LF87Hkz36shRp8@x', host='MRADBSERVER', database=mls_db) as mls:
                mls_schema_df = pd.read_sql_query(mls_query,mls)
                mls_schema_df.to_csv(path_name)

                return mls_schema_df   
            
        except Exception as err:
            print('error : %s' %err)

    with connector.connect(user='juanvasquez', password='LF87Hkz36shRp8@x', host='MRADBSERVER', port=3306, database='listings') as conn:
        query_str = "SELECT * FROM listings.mlses"
        query_df = pd.read_sql(query_str, conn)
        df = query_df[query_df.listing_table.notnull()]
        df.to_csv('listing_mlses.csv')
        id_df = df.loc[:,['id']]
        mls_db_df = df.loc[:,['id','listing_table']]
        lsiting_df = df.filter(regex='listing_')
        mls_dbs = ['mls_%i'%i for i in id_df.to_numpy()]
        
        result = [get_columns(id, table) for id, table in zip(df['id'], df['listing_table'])]
    
    
populate_data()