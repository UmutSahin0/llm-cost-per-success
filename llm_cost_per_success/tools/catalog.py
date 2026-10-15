"""Tool registry and non-oracle lexical discovery for the benchmark."""
from __future__ import annotations
import copy
from typing import Any

def tool(name: str, description: str, properties: dict[str, Any], terms: str) -> dict[str, Any]:
    return {"type":"function","name":name,"description":description,"parameters":{"type":"object","properties":properties,"required":list(properties)},"_retrieval_text":f"{name} {description} {terms}".lower()}

BASE_CATALOG = [
 tool("get_weather","Get forecast and precipitation probability for a city.",{"city":{"type":"string"}},"weather rain forecast temperature city"),
 tool("create_calendar_event","Create an event on a user's calendar.",{"title":{"type":"string"},"start":{"type":"string"}},"calendar meeting schedule appointment event"),
 tool("send_email","Send an email to a recipient.",{"to":{"type":"string"},"subject":{"type":"string"}},"email mail send recipient"),
 tool("web_search","Search the public web for current information.",{"query":{"type":"string"}},"web internet official documentation search"),
 tool("lookup_crm_account","Look up a company account and CRM activity.",{"company":{"type":"string"}},"crm customer account company sales meeting"),
 tool("create_invoice","Create an invoice in the billing system.",{"invoice_number":{"type":"string"}},"invoice billing payment bill create"),
 tool("get_invoice","Retrieve status and line items for an invoice.",{"invoice_number":{"type":"string"}},"invoice billing payment bill retrieve status"),
 tool("refund_invoice","Refund a paid invoice.",{"invoice_number":{"type":"string"}},"invoice billing payment refund reverse"),
 tool("track_shipment","Get shipment status by tracking number.",{"tracking_number":{"type":"string"}},"shipping shipment cargo tracking delivery"),
 tool("list_github_issues","List issues in a GitHub repository.",{"repository":{"type":"string"}},"github repository issue bug pull request"),
 tool("run_sql_query","Run a read-only SQL query against analytics data.",{"sql":{"type":"string"}},"sql database warehouse query analytics stock inventory"),
 tool("translate_text","Translate text to a requested target language.",{"text":{"type":"string"},"target_language":{"type":"string"}},"translate translation language german english"),
 tool("extract_pdf_text","Extract text from a PDF document.",{"path":{"type":"string"}},"pdf document contract extract clause text"),
 tool("send_slack_message","Post a message to a Slack channel.",{"channel":{"type":"string"},"message":{"type":"string"}},"slack channel message post notify"),
 tool("create_support_ticket","Create a customer support ticket.",{"title":{"type":"string"}},"support ticket customer complaint incident"),
 tool("get_product_inventory","Look up current inventory for a product SKU.",{"sku":{"type":"string"}},"product inventory sku stock quantity"),
 tool("query_knowledge_base","Search the internal knowledge base.",{"query":{"type":"string"}},"internal knowledge base handbook policy documentation"),
 tool("create_linear_issue","Create an issue in Linear.",{"title":{"type":"string"}},"linear issue bug task project"),
 tool("request_clarification","Ask the user for missing information before taking an action.",{"question":{"type":"string"}},"clarify clarification missing ambiguous ask question"),
 tool("respond_without_tool","Answer a writing or reasoning request that requires no external tool.",{"response":{"type":"string"}},"no tool write rewrite explain summarize"),
 # Realistic but unconnected integrations. They are deliberately unrelated (or
 # near-neighbours) so catalog scaling tests measure routing difficulty.
 tool("get_customer_profile","Retrieve a customer profile, preferences, and account status.",{"customer_id":{"type":"string"}},"customer profile user account preferences status"),
 tool("update_customer_profile","Update permitted fields on a customer profile.",{"customer_id":{"type":"string"},"updates":{"type":"object"}},"customer profile user account update preferences"),
 tool("search_contacts","Find people in the contacts directory by name, company, or role.",{"query":{"type":"string"}},"contacts people directory company role search"),
 tool("create_sales_lead","Create a sales lead and assign an initial owner.",{"company":{"type":"string"},"owner":{"type":"string"}},"sales lead prospect company owner create"),
 tool("list_sales_opportunities","List pipeline opportunities matching a sales stage or owner.",{"stage":{"type":"string"}},"sales opportunity pipeline deal stage owner list"),
 tool("update_sales_opportunity","Update amount, stage, or close date for a sales opportunity.",{"opportunity_id":{"type":"string"},"stage":{"type":"string"}},"sales opportunity pipeline deal stage update"),
 tool("create_quote","Prepare a sales quote for products and quantities.",{"account_id":{"type":"string"},"items":{"type":"array"}},"quote proposal sales pricing products quantities"),
 tool("get_subscription","Retrieve subscription plan, status, renewal date, and usage.",{"subscription_id":{"type":"string"}},"subscription plan renewal usage billing status"),
 tool("cancel_subscription","Cancel a subscription at the requested effective date.",{"subscription_id":{"type":"string"},"effective_date":{"type":"string"}},"subscription cancel billing renewal effective date"),
 tool("apply_promotion_code","Apply a promotion code to a customer billing account.",{"account_id":{"type":"string"},"code":{"type":"string"}},"promotion coupon discount billing account code"),

 tool("create_product","Create a product record in the product information system.",{"name":{"type":"string"},"sku":{"type":"string"}},"product catalog sku create item"),
 tool("update_product_price","Set a product price for a currency and sales region.",{"sku":{"type":"string"},"price":{"type":"number"}},"product sku price pricing update currency"),
 tool("list_product_reviews","List product ratings and written customer reviews.",{"sku":{"type":"string"}},"product review rating customer feedback list"),
 tool("create_purchase_order","Create a purchase order for a supplier.",{"supplier":{"type":"string"},"items":{"type":"array"}},"purchase order supplier procurement buy items"),
 tool("get_purchase_order","Retrieve a purchase order and fulfillment status.",{"purchase_order_id":{"type":"string"}},"purchase order procurement supplier fulfillment status"),
 tool("approve_expense_report","Approve or reject an employee expense report.",{"expense_report_id":{"type":"string"},"decision":{"type":"string"}},"expense report approve reject reimbursement employee"),
 tool("get_exchange_rate","Retrieve the current exchange rate for a currency pair.",{"base_currency":{"type":"string"},"quote_currency":{"type":"string"}},"exchange rate currency foreign conversion finance"),
 tool("create_budget","Create a department budget for a fiscal period.",{"department":{"type":"string"},"period":{"type":"string"}},"budget finance department fiscal planning create"),
 tool("get_budget_variance","Compare actual spend with a departmental budget.",{"department":{"type":"string"},"period":{"type":"string"}},"budget variance actual spend finance department"),
 tool("forecast_revenue","Generate a revenue forecast from pipeline and historical data.",{"period":{"type":"string"}},"revenue forecast sales finance prediction period"),

 tool("upload_file","Register an uploaded file and return its storage identifier.",{"filename":{"type":"string"},"content_type":{"type":"string"}},"file upload attachment storage document"),
 tool("list_files","List files in a workspace folder.",{"folder_id":{"type":"string"}},"file folder workspace storage list documents"),
 tool("move_file","Move a file to a different workspace folder.",{"file_id":{"type":"string"},"folder_id":{"type":"string"}},"file move folder workspace storage document"),
 tool("share_file","Grant a user or group access to a file.",{"file_id":{"type":"string"},"principal":{"type":"string"}},"file share permission access user group document"),
 tool("create_document","Create a collaborative document from a title and body.",{"title":{"type":"string"},"body":{"type":"string"}},"document create page note collaborative content"),
 tool("search_documents","Search workspace documents by text and metadata.",{"query":{"type":"string"}},"document workspace file search text metadata"),
 tool("summarize_document","Generate a concise summary of a stored document.",{"document_id":{"type":"string"}},"document summarize summary file content"),
 tool("extract_table_from_pdf","Extract tabular data from a PDF attachment.",{"file_id":{"type":"string"}},"pdf table extract spreadsheet document"),
 tool("create_dashboard","Create an analytics dashboard from named metrics.",{"title":{"type":"string"},"metrics":{"type":"array"}},"analytics dashboard metrics chart report create"),
 tool("schedule_report","Schedule recurring delivery of an analytics report.",{"report_id":{"type":"string"},"schedule":{"type":"string"}},"analytics report schedule recurring delivery email"),

 tool("create_user","Provision a user account in the identity directory.",{"email":{"type":"string"},"display_name":{"type":"string"}},"identity user account provision directory create"),
 tool("deactivate_user","Disable a user account and revoke active sessions.",{"user_id":{"type":"string"}},"identity user account deactivate disable access"),
 tool("assign_role","Assign an application role to a user.",{"user_id":{"type":"string"},"role":{"type":"string"}},"identity role permission user access assign"),
 tool("list_audit_events","List security and administrative audit events.",{"start_time":{"type":"string"}},"audit security event administration log list"),
 tool("create_api_key","Create a scoped API key for a service account.",{"service_account":{"type":"string"},"scope":{"type":"string"}},"api key token service account credential create"),
 tool("rotate_api_key","Rotate an existing API key and invalidate the prior key.",{"key_id":{"type":"string"}},"api key token credential rotate security"),
 tool("get_feature_flag","Retrieve the enabled state and rollout rules for a feature flag.",{"flag_key":{"type":"string"}},"feature flag rollout experiment enabled configuration"),
 tool("update_feature_flag","Update targeting or enabled state for a feature flag.",{"flag_key":{"type":"string"},"enabled":{"type":"boolean"}},"feature flag rollout experiment update enabled"),
 tool("create_webhook","Register a webhook endpoint for an event topic.",{"url":{"type":"string"},"event_type":{"type":"string"}},"webhook event endpoint integration create"),
 tool("list_webhooks","List registered webhook endpoints and delivery status.",{"event_type":{"type":"string"}},"webhook endpoint event delivery list integration"),

 tool("create_incident","Create an engineering incident with severity and service.",{"title":{"type":"string"},"severity":{"type":"string"}},"incident outage alert engineering severity service create"),
 tool("update_incident","Update incident status, commander, or mitigation notes.",{"incident_id":{"type":"string"},"status":{"type":"string"}},"incident outage alert engineering update status"),
 tool("list_on_call_schedule","Retrieve on-call shifts for an engineering service.",{"service":{"type":"string"}},"on call schedule pager service engineering shift"),
 tool("trigger_deployment","Start a deployment of an application version to an environment.",{"service":{"type":"string"},"environment":{"type":"string"}},"deployment deploy release application environment trigger"),
 tool("get_deployment_status","Retrieve deployment progress and health checks.",{"deployment_id":{"type":"string"}},"deployment deploy release status health checks"),
 tool("rollback_deployment","Roll an application deployment back to a prior version.",{"deployment_id":{"type":"string"}},"deployment rollback release application version"),
 tool("create_github_issue","Create an issue in a source-control repository.",{"repository":{"type":"string"},"title":{"type":"string"}},"github repository issue bug task create"),
 tool("create_pull_request","Open a pull request from a source branch to a target branch.",{"repository":{"type":"string"},"source_branch":{"type":"string"}},"github pull request code review branch create"),
 tool("get_ci_build_status","Retrieve build and test status for a commit or branch.",{"repository":{"type":"string"},"ref":{"type":"string"}},"ci build test pipeline github commit branch status"),
 tool("query_application_logs","Search application logs by service, time range, and text.",{"service":{"type":"string"},"query":{"type":"string"}},"logs observability application service search error"),

 tool("create_employee","Create an employee record in the HR system.",{"email":{"type":"string"},"legal_name":{"type":"string"}},"employee hr people hire record create"),
 tool("get_employee_directory","Search the employee directory by name, team, or location.",{"query":{"type":"string"}},"employee people directory team location search"),
 tool("request_time_off","Submit a paid-time-off request for an employee.",{"employee_id":{"type":"string"},"start_date":{"type":"string"}},"time off leave vacation pto employee request"),
 tool("approve_time_off","Approve or deny a pending time-off request.",{"request_id":{"type":"string"},"decision":{"type":"string"}},"time off leave vacation pto approve deny"),
 tool("create_job_posting","Create an open role in the recruiting system.",{"title":{"type":"string"},"department":{"type":"string"}},"job posting recruiting role hiring department create"),
 tool("list_job_candidates","List candidates for a job opening and their stage.",{"job_id":{"type":"string"}},"recruiting candidate applicant job interview list"),
 tool("schedule_interview","Schedule an interview with a candidate and panel.",{"candidate_id":{"type":"string"},"start":{"type":"string"}},"recruiting candidate interview schedule calendar panel"),
 tool("create_contract","Create a contract record for legal review.",{"counterparty":{"type":"string"},"title":{"type":"string"}},"legal contract agreement create review counterparty"),
 tool("request_contract_review","Send a contract to the legal review workflow.",{"contract_id":{"type":"string"}},"legal contract agreement review approval request"),
 tool("search_policy_library","Search corporate policies, procedures, and compliance guidance.",{"query":{"type":"string"}},"policy compliance procedure handbook internal search"),

]
def public_schema(item: dict[str, Any]) -> dict[str, Any]: return {key:value for key,value in item.items() if key!="_retrieval_text"}

def rich_parameters() -> dict[str, Any]:
    """Optional, realistic integration fields used only to enlarge JSON Schema."""
    return {
        "request_context":{"type":"object","description":"Caller identity, tenant scope, locale, timezone, and trace metadata used for authorization and audit logging.","properties":{"tenant_id":{"type":"string","description":"Stable tenant or workspace identifier."},"actor_id":{"type":"string","description":"Authenticated user or service principal identifier."},"locale":{"type":"string","enum":["en-US","tr-TR","de-DE","fr-FR"],"description":"Locale for text, dates, and user-facing formatting."},"timezone":{"type":"string","description":"IANA timezone used for interpreting relative timestamps."},"trace_id":{"type":"string","description":"Distributed tracing correlation identifier."}}},
        "execution_options":{"type":"object","description":"Controls validation, dry-run behavior, retry policy, and response expansion for this operation.","properties":{"dry_run":{"type":"boolean","description":"Validate the operation without committing external side effects."},"idempotency_key":{"type":"string","description":"Client-generated key that makes retried write operations safe."},"response_detail":{"type":"string","enum":["minimal","standard","expanded"],"description":"Amount of related resource data requested in the response."},"retry_policy":{"type":"string","enum":["none","safe_reads","transient_failures"],"description":"Retry behavior permitted for this request."}}},
        "filters":{"type":"array","description":"Optional structured predicates used to narrow records before the operation runs.","items":{"type":"object","properties":{"field":{"type":"string","description":"Canonical field name."},"operator":{"type":"string","enum":["eq","neq","contains","in","gte","lte"],"description":"Comparison operator."},"value":{"type":"string","description":"Comparison value serialized as text."}}}},
        "audit_metadata":{"type":"object","description":"Business reason and classification metadata retained with the operation for governance review.","properties":{"reason":{"type":"string","description":"Human-readable explanation for the requested operation."},"source_system":{"type":"string","description":"System or workflow that initiated the request."},"labels":{"type":"array","items":{"type":"string"},"description":"Free-form labels for reporting and audit searches."}}},
        "include_fields":{"type":"array","items":{"type":"string"},"description":"Named related resources and computed fields to include in the returned payload."},
        "sort":{"type":"array","description":"Ordered sorting rules for record-oriented responses.","items":{"type":"object","properties":{"field":{"type":"string","description":"Sortable canonical field."},"direction":{"type":"string","enum":["ascending","descending"],"description":"Requested sort direction."}}}},
        "page":{"type":"integer","minimum":1,"description":"One-based page number for paginated list responses."},
        "page_size":{"type":"integer","minimum":1,"maximum":500,"description":"Maximum number of records to return in a single page."},
        "effective_at":{"type":"string","format":"date-time","description":"Optional point-in-time timestamp for historical or scheduled operations."},
        "client_request_id":{"type":"string","description":"Caller-generated identifier retained for reconciliation and support investigations."},
        "attachments":{"type":"array","description":"Optional file references associated with this operation.","items":{"type":"object","properties":{"file_id":{"type":"string","description":"Opaque uploaded file identifier."},"filename":{"type":"string","description":"Original human-readable filename."},"content_type":{"type":"string","description":"IANA media type for the attached content."}}}},
        "notification":{"type":"object","description":"Optional post-operation notification policy.","properties":{"channels":{"type":"array","items":{"type":"string","enum":["email","slack","webhook"]},"description":"Destinations that should receive a completion notification."},"recipients":{"type":"array","items":{"type":"string"},"description":"Recipient identifiers for the selected notification channels."},"template":{"type":"string","description":"Named notification template to render after completion."}}},
    }

def inflate_schema(item: dict[str, Any]) -> dict[str, Any]:
    expanded=copy.deepcopy(item)
    expanded["parameters"]["additionalProperties"]=False
    expanded["parameters"]["properties"].update(rich_parameters())
    return expanded

def build_catalog(size: int, schema_profile: str="compact") -> list[dict[str, Any]]:
    if size < 1: raise ValueError("catalog size must be at least 1")
    items=list(BASE_CATALOG[:size])
    for index in range(len(BASE_CATALOG)+1,size+1):
        domain=("payments","shipping","crm","analytics","documents")[index%5]
        items.append(tool(f"archive_{domain}_record_{index}",f"Archive a {domain} record for compliance retention.",{"record_id":{"type":"string"}},f"archive {domain} record compliance retention"))
    if schema_profile=="compact": return items
    if schema_profile=="rich": return [inflate_schema(item) for item in items]
    raise ValueError(f"unknown schema profile: {schema_profile}")
