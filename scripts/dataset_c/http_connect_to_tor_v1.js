#!/usr/bin/env node
'use strict';
// QA-only HTTP CONNECT -> SOCKS5 bridge. No direct upstream sockets except 127.0.0.1:9050.
// DNS resolution of destinations is performed by Tor via SOCKS5 DOMAIN requests.
const net = require('net');
const HOST='127.0.0.1', PORT=18888, SOCKS_PORT=9050;
const log=(...args)=>console.log('BRIDGE',...args);
function fail(client, up, msg) {
  log('FAIL', msg);
  if (up && !up.destroyed) up.destroy();
  if (client && !client.destroyed) client.end('HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\nContent-Length: 0\r\n\r\n');
}
const server=net.createServer(client=>{
  client.setTimeout(25000,()=>client.destroy());
  let upstream, buf=Buffer.alloc(0), state='http', target='', port=0;
  function onClient(chunk){
    if(state!=='http')return;
    buf=Buffer.concat([buf,chunk]);
    if(buf.length>8192){fail(client,upstream,'http_request_too_long');return;}
    const idx=buf.indexOf('\r\n\r\n');
    if(idx<0)return;
    const line=buf.subarray(0,idx).toString('latin1').split('\r\n')[0];
    const match=/^CONNECT ([a-zA-Z0-9.-]{1,253}):(443) HTTP\/1\.[01]$/.exec(line);
    if(!match){fail(client,upstream,'invalid_CONNECT '+line.slice(0,90));return;}
    target=match[1];port=Number(match[2]);
    state='socks_hello';buf=Buffer.alloc(0);
    upstream=net.connect(SOCKS_PORT,HOST,()=>upstream.write(Buffer.from([5,1,0])));
    upstream.setTimeout(25000,()=>fail(client,upstream,'socks_timeout'));
    upstream.on('error',e=>fail(client,upstream,'socks_error '+e.code));
    upstream.on('close',()=>{if(!client.destroyed)client.destroy();});
    upstream.on('data',onSocks);
  }
  let q=Buffer.alloc(0);
  function onSocks(chunk){
    q=Buffer.concat([q,chunk]);
    if(state==='socks_hello'){
      if(q.length<2)return;
      if(q[0]!==5||q[1]!==0){fail(client,upstream,'bad_socks_auth');return;}
      q=q.subarray(2);state='socks_reply';
      const host=Buffer.from(target,'ascii');
      upstream.write(Buffer.concat([Buffer.from([5,1,0,3,host.length]),host,Buffer.from([port>>8,port&255])]));
    }
    if(state==='socks_reply'){
      if(q.length<5)return;
      let n;
      if(q[3]===1)n=10;
      else if(q[3]===4)n=22;
      else if(q[3]===3){if(q.length<5)return;n=7+q[4];}
      else {fail(client,upstream,'bad_socks_atyp');return;}
      if(q.length<n)return;
      if(q[0]!==5||q[1]!==0){fail(client,upstream,'socks_connect_status_'+q[1]);return;}
      const rest=q.subarray(n);q=Buffer.alloc(0);state='stream';
      log('TOR_SOCKS_CONNECT_OK',target+':'+port);
      upstream.removeListener('data',onSocks);
      client.removeListener('data',onClient);
      client.setTimeout(0);upstream.setTimeout(0);
      client.write('HTTP/1.1 200 Connection Established\r\n\r\n');
      if(rest.length)client.write(rest);
      client.pipe(upstream);upstream.pipe(client);
    }
  }
  client.on('data',onClient);
  client.on('error',()=>{if(upstream)upstream.destroy();});
  client.on('close',()=>{if(upstream)upstream.destroy();});
});
server.on('error',e=>{log('FATAL',e.message);process.exit(2);});
server.listen(PORT,HOST,()=>log('READY',HOST+':'+PORT,'UPSTREAM=127.0.0.1:'+SOCKS_PORT));
