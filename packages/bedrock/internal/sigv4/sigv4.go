package sigv4

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

type Credentials struct {
	Region          string
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
	Service         string
}

func Sign(req *http.Request, body []byte, creds Credentials, now time.Time) error {
	if creds.Service == "" {
		creds.Service = "bedrock"
	}
	if creds.Region == "" || creds.AccessKeyID == "" || creds.SecretAccessKey == "" {
		return fmt.Errorf("aws sigv4 requires region, access key id, and secret access key")
	}
	amzDate := now.UTC().Format("20060102T150405Z")
	date := now.UTC().Format("20060102")
	bodyHash := sha256Hex(body)
	req.Header.Set("X-Amz-Date", amzDate)
	req.Header.Set("X-Amz-Content-Sha256", bodyHash)
	if creds.SessionToken != "" {
		req.Header.Set("X-Amz-Security-Token", creds.SessionToken)
	}

	canonicalHeaders, signedHeaders := canonicalHeaders(req)
	canonicalRequest := strings.Join([]string{
		req.Method,
		canonicalURI(req.URL),
		canonicalQuery(req.URL),
		canonicalHeaders,
		signedHeaders,
		bodyHash,
	}, "\n")
	scope := strings.Join([]string{date, creds.Region, creds.Service, "aws4_request"}, "/")
	stringToSign := strings.Join([]string{
		"AWS4-HMAC-SHA256",
		amzDate,
		scope,
		sha256Hex([]byte(canonicalRequest)),
	}, "\n")
	signingKey := signingKey(creds.SecretAccessKey, date, creds.Region, creds.Service)
	signature := hex.EncodeToString(hmacSHA256(signingKey, stringToSign))
	req.Header.Set("Authorization", fmt.Sprintf("AWS4-HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s", creds.AccessKeyID, scope, signedHeaders, signature))
	return nil
}

func canonicalURI(u *url.URL) string {
	if u.EscapedPath() == "" {
		return "/"
	}
	return u.EscapedPath()
}

func canonicalQuery(u *url.URL) string {
	q := u.Query()
	keys := make([]string, 0, len(q))
	for key := range q {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var parts []string
	for _, key := range keys {
		values := q[key]
		sort.Strings(values)
		for _, value := range values {
			parts = append(parts, url.QueryEscape(key)+"="+url.QueryEscape(value))
		}
	}
	return strings.Join(parts, "&")
}

func canonicalHeaders(req *http.Request) (string, string) {
	headers := map[string]string{"host": req.URL.Host}
	for key, values := range req.Header {
		lower := strings.ToLower(key)
		if len(values) == 0 {
			continue
		}
		headers[lower] = strings.Join(values, ",")
	}
	keys := make([]string, 0, len(headers))
	for key := range headers {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var b strings.Builder
	for _, key := range keys {
		b.WriteString(key)
		b.WriteByte(':')
		b.WriteString(strings.Join(strings.Fields(headers[key]), " "))
		b.WriteByte('\n')
	}
	return b.String(), strings.Join(keys, ";")
}

func signingKey(secret, date, region, service string) []byte {
	kDate := hmacSHA256([]byte("AWS4"+secret), date)
	kRegion := hmacSHA256(kDate, region)
	kService := hmacSHA256(kRegion, service)
	return hmacSHA256(kService, "aws4_request")
}

func hmacSHA256(key []byte, value string) []byte {
	h := hmac.New(sha256.New, key)
	_, _ = h.Write([]byte(value))
	return h.Sum(nil)
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
