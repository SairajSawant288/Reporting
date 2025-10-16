import pandas as pd
import numpy as np
import logging
from pathlib import Path


class LoanPortfolioReportAutomationPandas:
    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory)
        self.logger = self.setup_logging()
        self.output_report = None

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        return logging.getLogger('LoanPortfolioAutomationPandas')

    def load_all_dataframes(self):
        self.logger.info("Loading all data CSV files into DataFrames")
        self.aum_month_report = pd.read_csv(self.data_directory / "aum_month_report.csv")
        self.sharing_ratio_master = pd.read_csv(self.data_directory / "sharing_ratio_master.csv")
        self.bookdbet_tagging = pd.read_csv(self.data_directory / "bookdbet_tagging.csv")
        self.bookdbet_tagging_product = pd.read_csv(self.data_directory / "bookdbet_tagging_product.csv")
        self.write_off_lans = pd.read_csv(self.data_directory / "write_off_lans.csv")
        self.morat_data = pd.read_csv(self.data_directory / "morat_data.csv")
        self.foreclosure_tagging = pd.read_csv(self.data_directory / "foreclosure_tagging.csv")
        self.cal_managed_da_ptc_master = pd.read_csv(self.data_directory / "CAL_MANAGED_DA_PTC_Master.csv")
        self.overdue_report = pd.read_csv(self.data_directory / "overdue_report.csv")
        self.portfolio_report_opening = pd.read_csv(self.data_directory / "portfolio_report_opening.csv")
        self.repo_stock_report = pd.read_csv(self.data_directory / "Repo_Stock_report.csv")
        self.co_lending_rate_master_report = pd.read_csv(self.data_directory / "co_lending_rate_master_report.csv")

    def create_output_report_pandas(self):
        self.logger.info("Creating output_report DataFrame in pandas")
        self.bookdbet_tagging_product.rename(columns={'LOAN ACCOUNT NO.':'LOAN_ACCOUNT_NO'},inplace=True)
        
        df = pd.merge(self.aum_month_report, 
                      self.bookdbet_tagging_product[['LOAN_ACCOUNT_NO', 'PRODUCT']],
                      on='LOAN_ACCOUNT_NO', how='left')
        df.drop(columns=['PRODUCT_x','TYPE'],inplace=True)
        df.rename(columns={'PRODUCT_y':'PRODUCT'},inplace=True)
        

        def classify_type(row):
            product = row['PRODUCT']
            if product in ('DDF','DDF-New','DDF-Used','DDF-E Auto','DDF-NEW'):
                return 'Wholesale'
            st = row.get('SUBROGATION_TAGGING', '')
            # bm = row.get('BUSINESSMODEL', '')
            co = row.get('COLENDER', '')
            # if st == 'ICICI Subrogation':
            #     return 'ICICI Subrogation'
            # if bm == 'ICICI':
            #     return 'ICICI Co-lending'
            if st == 'Muthoot FLDG':
                return 'Muthoot FLDG'
            if st == 'Vivriti FLDG':
                return 'Vivriti FLDG'
            if st == 'Poonawala FLDG':
                return 'Poonawala FLDG'
            if co == 'Greaves co-lending':
                return 'Greaves Co-lending'
            if co == 'Muthoot Capital Co-lending':
                return 'Muthoot Capital Co-lending'
            if co == 'Vivriti Capital Co-lending':
                return 'Vivriti Capital Co-lending'
            if co == 'Poonawala Fincorp Co-lending':
                return 'Poonawala Fincorp Co-lending'
            if product == 'EV Insti':
                return 'Institutional Underwriting - Loan'
            return 'Retail'
        
        df['TYPE'] = df.apply(classify_type, axis=1)
        #Test
        # print(df.columns)
        # df.to_csv("test.csv",index=False)

        df['unbilled_adv_emi'] = df.get('GROSS_ADVANCE_INSTL_AMT', 0).fillna(0) - df.get('GROSS_BILLED_ADVANCE_INSTL_AMT', 0).fillna(0)
        df['total_osp_from_finone'] = df.get('UNBILLED_PRINCIPAL_AMOUNT', 0)

        
        # Join tagging
        tagging = self.bookdbet_tagging.rename(columns={'LAN': 'LOAN_ACCOUNT_NO','DA / PTC TRANCH WISE - BOOKDEBT':'DA_PTC_TRANCH_WISE_BOOKDEBT','DA / PTC TRANCH WISE - AUM':'DA_PTC_TRANCH_WISE_AUM','Managed DA / PTC':'MANAGED_DA_PTC'})[
            ['LOAN_ACCOUNT_NO', 'DA_PTC_TRANCH_WISE_BOOKDEBT', 'DA_PTC_TRANCH_WISE_AUM', 'MANAGED_DA_PTC']]
        df = pd.merge(df, tagging, on='LOAN_ACCOUNT_NO', how='left')

        
        # Join write-off info
        
        write_off_renamed = self.write_off_lans.rename(columns={'LAN': 'LOAN_ACCOUNT_NO', 'WRITE OFF': 'write_off_status', 'Month': 'write_off_date'})
        df = pd.merge(df, write_off_renamed[['LOAN_ACCOUNT_NO', 'write_off_status', 'write_off_date']], on='LOAN_ACCOUNT_NO', how='left')
        df['write_off_status'] = df['write_off_status'].fillna('NO')
        # print(df)
        self.output_report = df
        self.logger.info("output_report DataFrame created")

    def process_full_business_logic_pandas(self):
        df = self.output_report
        srm = self.sharing_ratio_master
        amr = self.aum_month_report
        morat = self.morat_data

        df['osp_final'] = np.where(df['MANAGED_DA_PTC'] == 'Managed - ARC', 0, df['total_osp_from_finone'])
        df['osp_final_excluding_write_off'] = df['osp_final'].copy()
        df.loc[df['write_off_status'] == 'YES', 'osp_final_excluding_write_off'] = 0

        overdue_dict = amr.set_index('LOAN_ACCOUNT_NO')['OVERDUE_PRIN_AMT'].to_dict()
        df['OVERDUE_PRIN_AMT'] = df['LOAN_ACCOUNT_NO'].map(overdue_dict)
        df['overdue_prin_amt_excluding_write_off'] = df['OVERDUE_PRIN_AMT'].copy()
        df.loc[df['MANAGED_DA_PTC'] == 'Managed - ARC', 'overdue_prin_amt_excluding_write_off'] = 0
        df.loc[df['write_off_status'] == 'YES', 'overdue_prin_amt_excluding_write_off'] = 0
        df['overdue_prin_amt_excluding_write_off'] = np.where(df['overdue_prin_amt_excluding_write_off'] < 0,0,df['overdue_prin_amt_excluding_write_off'])

        srm.rename(columns={'DA / PTC Tranch wise - AUM':'DA_PTC_TRANCH_WISE_AUM','Own Share':'OWN_SHARE'},inplace=True)
        def calc_principal_od_own(row):
            # Lookup OWN_SHARE for the row’s type and DA_PTC_TRANCH_WISE_AUM
            type_match = srm.loc[srm['DA_PTC_TRANCH_WISE_AUM'] == row['TYPE'], 'OWN_SHARE']
            daptc_match = srm.loc[srm['DA_PTC_TRANCH_WISE_AUM'] == row['DA_PTC_TRANCH_WISE_AUM'], 'OWN_SHARE']

            # If lookup yields no rows, treat share as 0
            type_share = float(type_match.iloc[0]) if not type_match.empty and pd.notna(type_match.iloc[0]) else 0.0
            daptc_share = float(daptc_match.iloc[0]) if not daptc_match.empty and pd.notna(daptc_match.iloc[0]) else 0.0

            overdue = row.get('overdue_prin_amt_excluding_write_off', 0.0) or 0.0


            # If overdue is missing or zero, result is zero
            if overdue <= 0:
                return 0.0

            # Compute combined percentage
            type_per = type_share / 100.0
            daptc_per = daptc_share / 100.0


            # If either percentage is zero, result is zero
            if type_per == 0.0 and daptc_per == 0.0:
                return overdue

            # Final calculation
            result = 0
            if daptc_per > 0 and type_per > 0:
                result = round(overdue * type_per * daptc_per,2)
                return max(result, 0.0)
            if type_per > 0:
                result = round(overdue * type_per,2)
                return max(result, 0.0)
            if daptc_per > 0:
                result = round(overdue * daptc_per,2)
                return max(result, 0.0)

            # Ensure non-negative
            

        df['principal_od_own_from_par'] = df.apply(calc_principal_od_own, axis=1)
        
        morat_dict = morat.set_index('LOAN_ACCOUNT_NO')['MORAT'].to_dict()
        df['morat_principal_from_tab'] = df['LOAN_ACCOUNT_NO'].map(morat_dict).fillna(0)
        df.loc[df['write_off_status'] == 'YES', 'morat_principal_from_tab'] = 0
        df.loc[df['MANAGED_DA_PTC'] == 'Managed - ARC', 'morat_principal_from_tab'] = 0
        # print(df.info())
        df['aum_including_written_off'] = df['osp_final_excluding_write_off'].fillna(0) + df['principal_od_own_from_par'].fillna(0) + df['morat_principal_from_tab'].fillna(0)

        df['write_off_amt'] = 0
        df['aum_excluding_written_off'] = df['aum_including_written_off'].copy()
        # df.loc[df['write_off_amt'] > 0, 'aum_excluding_written_off'] = 0
        df.loc[df['write_off_status'] == 'YES', 'aum_excluding_written_off'] = 0
        df.loc[df['MANAGED_DA_PTC'] == 'Managed - ARC', 'aum_excluding_written_off'] = 0

        # Additional logic similar to above SQL updates can be added here
        # Vishal concern 
        df['PRINCIPAL_OD_Managed'] = df['overdue_prin_amt_excluding_write_off'] - df['principal_od_own_from_par']

        df['AUM_including_advance_EMI'] = df['aum_excluding_written_off'] + df['PRINCIPAL_OD_Managed']
        df.loc[df['write_off_status'] == 'YES', 'AUM_including_advance_EMI'] = 0
        df.loc[df['MANAGED_DA_PTC'] == 'Managed - ARC', 'AUM_including_advance_EMI'] = 0

        df['ADVANCE_EMI'] = df['unbilled_adv_emi'].copy()
        df.loc[df['write_off_status'] == 'YES', 'ADVANCE_EMI'] = 0
        df.loc[df['MANAGED_DA_PTC'] == 'Managed - ARC', 'ADVANCE_EMI'] = 0

        df['Final_AUM'] = df['AUM_including_advance_EMI'] - df['ADVANCE_EMI']
        df.loc[df['write_off_status'] == 'YES', 'Final_AUM'] = 0
        df.loc[df['MANAGED_DA_PTC'] == 'Managed - ARC', 'Final_AUM'] = 0
        df['Final_AUM'] = np.where(df['Final_AUM'] < 0,0,df['Final_AUM'])

        df['OSP_Final_Min_ADVANCE_EMI'] = df['osp_final_excluding_write_off'] - df['ADVANCE_EMI']
        df.loc[df['write_off_status'] == 'YES', 'OSP_Final_Min_ADVANCE_EMI'] = 0
        df.loc[df['MANAGED_DA_PTC'] == 'Managed - ARC', 'OSP_Final_Min_ADVANCE_EMI'] = 0

        # Future owned shares
        def calc_future_owned_shares(row):
            # print("row")
            # print(row)
            # Lookup OWN_SHARE for the row’s type and DA_PTC_TRANCH_WISE_AUM
            type_match = srm.loc[srm['DA_PTC_TRANCH_WISE_AUM'] == row['TYPE'], 'OWN_SHARE']
            daptc_match = srm.loc[srm['DA_PTC_TRANCH_WISE_AUM'] == row['DA_PTC_TRANCH_WISE_AUM'], 'OWN_SHARE']
            # print("Type match",type_match)
            # print("daptc match",daptc_match)
            # If lookup yields no rows, treat share as 0
            type_share = float(type_match.iloc[0]) if not type_match.empty and pd.notna(type_match.iloc[0]) else 0.0
            daptc_share = float(daptc_match.iloc[0]) if not daptc_match.empty and pd.notna(daptc_match.iloc[0]) else 0.0

            # print("Type share",type_share)
            # print("daptc share",daptc_share)
            overdue = row.get('OSP_Final_Min_ADVANCE_EMI', 0.0) or 0.0

            # print("overdue",overdue)
            # If overdue is missing or zero, result is zero
            if overdue <= 0:
                return 0.0

            # Compute combined percentage
            type_per = type_share / 100.0
            daptc_per = daptc_share / 100.0

            # print("Type per",type_per)
            # print("daptc per",daptc_per)
            # If either percentage is zero, result is zero
            if type_per == 0.0 and daptc_per == 0.0:
                return overdue

            # Final calculation
            result = 0
            if daptc_per > 0 and type_per > 0:
                result = round(overdue * type_per * daptc_per,2)
                return max(result, 0.0)
            if type_per > 0:
                result = round(overdue * type_per,2)
                return max(result, 0.0)
            if daptc_per > 0:
                result = round(overdue * daptc_per,2)
                return max(result, 0.0)
            

        # df_ = df[(df['LOAN_ACCOUNT_NO'] == 'MCOMCN000005240049') | (df['LOAN_ACCOUNT_NO'] == 'JANN2W000005034582') | (df['LOAN_ACCOUNT_NO'] == 'MUMVCU000005283597')]
        df['FUTURE_OWNED_SHARE'] = df.apply(calc_future_owned_shares,axis=1)
        # df_ = df[(df['LOAN_ACCOUNT_NO'] == 'MCOMCN000005240049') | (df['LOAN_ACCOUNT_NO'] == 'JANN2W000005034582') | (df['LOAN_ACCOUNT_NO'] == 'MUMVCU000005283597')]

        df['FUTURE_MANAGED_SHARE'] = df['OSP_Final_Min_ADVANCE_EMI'] - df['FUTURE_OWNED_SHARE']

        # MORAT_OWN_AMOUNT
        def calc_morat_own_amount(row):
            # print("row")
            # print(row)
            # Lookup OWN_SHARE for the row’s type and DA_PTC_TRANCH_WISE_AUM
            type_match = srm.loc[srm['DA_PTC_TRANCH_WISE_AUM'] == row['TYPE'], 'OWN_SHARE']
            daptc_match = srm.loc[srm['DA_PTC_TRANCH_WISE_AUM'] == row['DA_PTC_TRANCH_WISE_AUM'], 'OWN_SHARE']
            # print("Type match",type_match)
            # print("daptc match",daptc_match)
            # If lookup yields no rows, treat share as 0
            type_share = float(type_match.iloc[0]) if not type_match.empty and pd.notna(type_match.iloc[0]) else 0.0
            daptc_share = float(daptc_match.iloc[0]) if not daptc_match.empty and pd.notna(daptc_match.iloc[0]) else 0.0

            # print("Type share",type_share)
            # print("daptc share",daptc_share)
            overdue = row.get('morat_principal_from_tab', 0.0) or 0.0

            # print("overdue",overdue)
            # If overdue is missing or zero, result is zero
            if overdue <= 0:
                return 0.0

            # Compute combined percentage
            type_per = type_share / 100.0
            daptc_per = daptc_share / 100.0

            # print("Type per",type_per)
            # print("daptc per",daptc_per)
            # If either percentage is zero, result is zero
            if type_per == 0.0 and daptc_per == 0.0:
                # print("Inside Overdue")
                return overdue

            # Final calculation
            result = 0
            if daptc_per > 0 and type_per > 0:
                result = round(overdue * type_per * daptc_per,2)
                return max(result, 0.0)
            if type_per > 0:
                result = round(overdue * type_per,2)
                return max(result, 0.0)
            if daptc_per > 0:
                result = round(overdue * daptc_per,2)
                return max(result, 0.0)
            
        # df = df[(df['LOAN_ACCOUNT_NO'] == 'BHRN2WA002044') ]
        # | (df['LOAN_ACCOUNT_NO'] == 'JANN2W000005034582') | (df['LOAN_ACCOUNT_NO'] == 'MUMVCU000005283597')]
        df['MORAT_OWN_AMOUNT'] = df.apply(calc_morat_own_amount,axis=1)

        df['MORAT_MANAGED_AMOUNT'] = df['morat_principal_from_tab'] - df['MORAT_OWN_AMOUNT']

        df['TOTAL_AUM_OWNED_SHARE'] = df['principal_od_own_from_par'] + df['FUTURE_OWNED_SHARE'] + df['MORAT_OWN_AMOUNT']

        df['TOTAL_AUM_MANAGED_SHARE'] = df['PRINCIPAL_OD_Managed'] + df['FUTURE_MANAGED_SHARE'] + df['MORAT_MANAGED_AMOUNT']
        df = df[df['write_off_status'] != 'YES']

        df.columns = df.columns.str.upper()
        self.foreclosure_tagging.rename(columns={'LANs':'LOAN_ACCOUNT_NO'},inplace=True)
        df = df[~df['LOAN_ACCOUNT_NO'].isin(self.foreclosure_tagging['LOAN_ACCOUNT_NO'])]

        # 15th changes
        df = pd.merge(df, self.overdue_report, left_on='LOAN_ACCOUNT_NO',right_on='AGREEMENT_NO', how='left')
        df['CLOSING_EMI_OVERDUE'] = df['EMI_OVERDUE'].copy()

        #Testing Remaining
        df['MONTH_END_DPD'] = np.where(df['CLOSING_EMI_OVERDUE'] > 0,df['DPD'],'')

        df = pd.merge(df, self.portfolio_report_opening, on='LOAN_ACCOUNT_NO', how='left')
        # df.rename(columns={'ASSET_CLASSIFICATION':'ASSET_CLASSIFICATION_FINONE'},inplace=True)
        df['ASSET_CLASSIFICATION_FINONE'] = np.where(df['CLOSING_EMI_OVERDUE'] > 0,df['ASSET_CLASSIFICATION'],'')
        df['DATE_OF_MATURITY'] = df['MATURITYDATE'].copy()
        

        self.output_report = df
        self.logger.info("Business logic processing completed")

    def export_output_report(self, report_directory='loan_portfolio_report.csv'):
        self.output_report.to_csv(report_directory, index=False)
        self.logger.info(f"Exported output_report as {report_directory}")




if __name__ == "__main__":
    data_dir = "/Users/sairajsawant/Desktop/work/Vishal/code/data" # Update this path accordingly

    automation = LoanPortfolioReportAutomationPandas(data_dir)
    automation.load_all_dataframes()
    automation.create_output_report_pandas()
    automation.process_full_business_logic_pandas()
    automation.export_output_report("/Users/sairajsawant/Desktop/work/Vishal/code/Generated_Report/loan_portfolio_report.csv")

    print("Loan portfolio report generated successfully.")
