@echo off
rem -------------------------------------------------
rem Pinterest Scraper – Apify CLI
rem -------------------------------------------------

:: 1 Log in once (uncomment if you want to force login each run)
:: apify login

:: 2 Ensure a search‑terms file exists (creates demo if missing)
if not exist pinterest_search_terms.md (
echo summer fashion > pinterest_search_terms.md
echo home decor ideas >> pinterest_search_terms.md
echo vintage furniture >> pinterest_search_terms.md
    )

:: 3 Run the actor – inline JSON version
apify call fatihtahta/pinterest-scraper-search --input "{\"searchTermsFile\":\"pinterest_search_terms.md\"}"

pause