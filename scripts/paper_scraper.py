import os
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import csv
import time

def search_arxiv(query, max_results=50):
    """Search ArXiv for research papers matching the query."""
    print(f"Searching ArXiv for: {query}")
    
    # URL encode the query
    encoded_query = urllib.parse.quote(query)
    url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    
    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        
        papers = []
        # ArXiv XML namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            published = entry.find('atom:published', ns).text
            authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
            
            # Get PDF link
            pdf_link = ""
            for link in entry.findall('atom:link', ns):
                if link.attrib.get('title') == 'pdf':
                    pdf_link = link.attrib.get('href')
                    break
                    
            papers.append({
                "title": title,
                "authors": ", ".join(authors),
                "published_date": published,
                "summary": summary,
                "pdf_url": pdf_link
            })
            
        return papers
    except Exception as e:
        print(f"Error fetching from ArXiv: {e}")
        return []

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scraper_dir = os.path.join(base_dir, 'research_papers')
    os.makedirs(scraper_dir, exist_ok=True)
    os.makedirs(os.path.join(scraper_dir, 'pdfs'), exist_ok=True)
    
    # Define our specific niche queries
    queries = [
        'all:"fabric defect detection"',
        'all:"textile anomaly detection"',
        'all:"fabric inspection" AND all:"deep learning"',
        'all:"vision transformer" AND all:"defect detection"'
    ]
    
    all_papers = []
    seen_titles = set()
    
    for query in queries:
        papers = search_arxiv(query, max_results=30)
        for p in papers:
            if p['title'] not in seen_titles:
                all_papers.append(p)
                seen_titles.add(p['title'])
        time.sleep(3) # Be polite to ArXiv API
        
    print(f"Found {len(all_papers)} unique research papers in this niche.")
    
    # Save metadata to CSV
    csv_path = os.path.join(scraper_dir, 'fabric_defect_papers.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["title", "authors", "published_date", "summary", "pdf_url"])
        writer.writeheader()
        writer.writerows(all_papers)
        
    print(f"Saved metadata to {csv_path}")
    
    # Optionally download the top 5 PDFs as a sample
    print("Downloading Top 5 PDFs...")
    for i, paper in enumerate(all_papers[:5]):
        if paper['pdf_url']:
            pdf_filename = os.path.join(scraper_dir, 'pdfs', f"Paper_{i+1}.pdf")
            try:
                urllib.request.urlretrieve(paper['pdf_url'] + '.pdf', pdf_filename)
                print(f"Downloaded: {paper['title'][:50]}...")
                time.sleep(2)
            except Exception as e:
                print(f"Failed to download PDF {i+1}: {e}")

if __name__ == "__main__":
    main()
