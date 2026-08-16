const target = `http://${process.env.FASTAPI_HOST || '127.0.0.1'}:${process.env.FASTAPI_PORT || 5000}`;

module.exports = {
  '/api': {
    target,
    secure: false,
    changeOrigin: true,
    timeout: 600000,
    proxyTimeout: 600000,
  },
};
