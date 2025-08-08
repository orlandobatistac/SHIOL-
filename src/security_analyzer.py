
#!/usr/bin/env python3
"""
Custom Security Analyzer for SHIOL+ Project
Scans for common security vulnerabilities
"""

import os
import re
import json
from typing import Dict, List, Any
from datetime import datetime

class SecurityAnalyzer:
    def __init__(self):
        self.issues = []
        self.severity_colors = {
            "CRITICAL": "\033[91m🚨",
            "HIGH": "\033[93m⚠️",
            "MEDIUM": "\033[94mℹ️",
            "LOW": "\033[92m✓"
        }
    
    def scan_hardcoded_secrets(self, file_path: str):
        """Scan for hardcoded passwords and secrets"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                lines = content.split('\n')
                
                patterns = [
                    (r'password\s*=\s*["\']([^"\']+)["\']', 'Hardcoded password'),
                    (r'api_key\s*=\s*["\']([^"\']+)["\']', 'Hardcoded API key'),
                    (r'secret\s*=\s*["\']([^"\']+)["\']', 'Hardcoded secret'),
                    (r'token\s*=\s*["\']([^"\']+)["\']', 'Hardcoded token')
                ]
                
                for i, line in enumerate(lines):
                    for pattern, description in patterns:
                        matches = re.finditer(pattern, line, re.IGNORECASE)
                        for match in matches:
                            if len(match.group(1)) > 5:  # Avoid false positives
                                self.issues.append({
                                    'severity': 'CRITICAL',
                                    'file': file_path,
                                    'line': i + 1,
                                    'issue': description,
                                    'evidence': line.strip(),
                                    'recommendation': 'Move to environment variables'
                                })
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
    
    def scan_cors_issues(self, file_path: str):
        """Scan for CORS misconfigurations"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                if 'allow_origins=["*"]' in content or "allow_origins=['*']" in content:
                    self.issues.append({
                        'severity': 'CRITICAL',
                        'file': file_path,
                        'line': 'Multiple',
                        'issue': 'Permissive CORS configuration',
                        'evidence': 'allow_origins=["*"]',
                        'recommendation': 'Restrict to specific domains'
                    })
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
    
    def scan_directory(self, directory: str):
        """Scan entire directory for security issues"""
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.py', '.js', '.html')):
                    file_path = os.path.join(root, file)
                    self.scan_hardcoded_secrets(file_path)
                    if file.endswith('.py'):
                        self.scan_cors_issues(file_path)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate security report"""
        severity_counts = {}
        for issue in self.issues:
            severity = issue['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'scan_date': datetime.now().isoformat(),
            'total_issues': len(self.issues),
            'severity_breakdown': severity_counts,
            'issues': self.issues
        }
    
    def print_report(self):
        """Print colored security report"""
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("🛡️  SHIOL+ SECURITY ANALYSIS REPORT")
        print("="*60)
        print(f"📅 Scan Date: {report['scan_date']}")
        print(f"📊 Total Issues: {report['total_issues']}")
        
        if report['severity_breakdown']:
            print("\n📈 Severity Breakdown:")
            for severity, count in report['severity_breakdown'].items():
                color = self.severity_colors.get(severity, "")
                print(f"  {color} {severity}: {count}\033[0m")
        
        if self.issues:
            print("\n🔍 Detailed Issues:")
            for i, issue in enumerate(self.issues, 1):
                color = self.severity_colors.get(issue['severity'], "")
                print(f"\n{i}. {color} {issue['severity']}\033[0m - {issue['issue']}")
                print(f"   📁 File: {issue['file']}:{issue['line']}")
                print(f"   🔍 Evidence: {issue['evidence']}")
                print(f"   💡 Recommendation: {issue['recommendation']}")
        else:
            print("\n✅ No security issues detected!")

def main():
    analyzer = SecurityAnalyzer()
    analyzer.scan_directory('src/')
    analyzer.print_report()
    
    # Save JSON report
    report = analyzer.generate_report()
    with open('security_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Full report saved to: security_report.json")

if __name__ == "__main__":
    main()
