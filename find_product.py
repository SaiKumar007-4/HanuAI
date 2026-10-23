from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth
import time

def find_product_with_reviews():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="msedge",
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-infobars"
            ]
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
            viewport={"width": 1366, "height": 768},
            locale="en-CA",
            timezone_id="America/Toronto"
        )
        page = context.new_page()
        Stealth().apply_stealth_sync(page)
        
        try:
            print("Searching for popular products...")
            page.goto("https://www.bestbuy.ca/en-ca/search?search=car+seat", timeout=30000, wait_until="domcontentloaded")
            time.sleep(4)
            print("Title:", page.title())
            
            # Find product links
            links = page.query_selector_all("a[href*='/en-ca/product/']")
            print(f"Found {len(links)} product links:")
            seen = set()
            for l in links:
                href = l.get_attribute("href")
                if href and href not in seen and "/product/" in href:
                    seen.add(href)
                    text = l.inner_text().strip().replace("\n", " ")
                    if text:
                        print(f"Link: https://www.bestbuy.ca{href} | Text: {text[:60]}")
                if len(seen) >= 8:
                    break
        except Exception as e:
            print("Error:", e)
        finally:
            browser.close()

if __name__ == "__main__":
    find_product_with_reviews()
