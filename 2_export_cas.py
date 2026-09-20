# 1
#SAS.df2sd(
#    clean_df,
#    "WORK.READMIT_CLEAN"
#)

# 2
#SAS.submit("""
#proc contents data=WORK.READMIT_CLEAN;
#run;

#proc sql;
#    select count(*) as N_ROWS
#    from WORK.READMIT_CLEAN;
#quit;
#""")

# 3
#SAS.submit("""
#proc casutil;
#
#    /* Remove the previous CAS table if it exists */
#    droptable
#        incaslib="CASUSER"
#        casdata="READMIT_CLEAN"
#        quiet;
#
#    /* Upload the cleaned table and make it available globally */
#    load data=WORK.READMIT_CLEAN
#         outcaslib="CASUSER"
#         casout="READMIT_CLEAN"
#         promote;
#
#quit;
#""")

# 4
SAS.submit("""
proc casutil;
    list tables incaslib="CASUSER";
quit;
""")